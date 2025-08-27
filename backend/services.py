# services.py
import os
import re
import traceback
from io import BytesIO
from typing import List, Dict, Any

import PyPDF2
import torch
from nltk.tokenize import sent_tokenize
from sentence_transformers import util

# Import configurations and models from db_config.py
from db_config import (
    documents_collection, pc_index, embedding_tokenizer, embedding_model,
    reranking_model, gemini_model, EMB_DIM
)

# --- CORRECTED IMPORTS ---
# Import your custom utility functions directly
from utils.extract_summary import make_summary
from utils.gpt_text_generation import get_title_date_parties, get_doc_based_judgement
# Correctly import the 'update' module from the 'helpers' directory
from helpers import update as custom_update_helpers


# --- Constants ---
CHUNK_TOKENS = 400
CHUNK_OVERLAP = 50

# --- Helper Function ---
def sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitizes a metadata dictionary for Pinecone by converting all values
    to compatible types (string, number, boolean, or list of strings).
    """
    sanitized = {}
    for k, v in meta.items():
        if v is None:
            sanitized[k] = ""
        elif isinstance(v, list):
            sanitized[k] = [str(item) for item in v]
        elif isinstance(v, (str, int, float, bool)):
            sanitized[k] = v
        else:
            sanitized[k] = str(v)
    return sanitized

# --- Embedding Generation ---
@torch.inference_mode()
def get_embedding(text: str) -> List[float]:
    if not text: text = ""
    inputs = embedding_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = embedding_model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().tolist()
    if len(emb) != EMB_DIM:
        emb = (emb + [0.0] * EMB_DIM)[:EMB_DIM]
    return emb

# --- Text Chunking Logic ---
def _split_by_markers(text: str) -> List[str]:
    parts = re.split(
        r'(?i)(?=section\s+\d+[a-z]*\b|sec\.\s*\d+[a-z]*\b|article\s+\d+\b|para\.\s*\d+\b)',
        text
    )
    return [p.strip() for p in parts if p.strip()] or [text]

def smart_split_long_section(section: str) -> List[str]:
    sentences = sent_tokenize(section)
    chunks, current_chunk, current_len = [], [], 0
    for s in sentences:
        s_tokens = embedding_tokenizer.encode(s, add_special_tokens=False)
        if current_len + len(s_tokens) > CHUNK_TOKENS and current_chunk:
            chunks.append(" ".join(current_chunk).strip())
            prev_chunk_text = " ".join(current_chunk)
            prev_tokens = embedding_tokenizer.encode(prev_chunk_text, add_special_tokens=False)
            overlap_text = embedding_tokenizer.decode(prev_tokens[-CHUNK_OVERLAP:])
            current_chunk = [overlap_text, s]
            current_len = len(embedding_tokenizer.encode(overlap_text + s, add_special_tokens=False))
        else:
            current_chunk.append(s)
            current_len += len(s_tokens)
    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())
    return chunks

def chunk_text(text: str) -> List[str]:
    sections = _split_by_markers(text)
    all_chunks = []
    for sec in sections:
        sec_tokens = embedding_tokenizer.encode(sec, add_special_tokens=False)
        if len(sec_tokens) <= CHUNK_TOKENS:
            all_chunks.append(sec)
        else:
            all_chunks.extend(smart_split_long_section(sec))
    return [c for c in all_chunks if c and not c.isspace()]

# --- Gemini Fallback Function ---
def get_gemini_fallback_answer(query: str) -> str:
    print("[FALLBACK] Querying Gemini Flash API with enhanced prompt...")
    try:
        # This new prompt guides the model to provide more structured and grounded answers.
        prompt = f"""
        You are an expert legal assistant specializing in Indian law. Your task is to answer the user's query based on your general knowledge.

        **Response Guidelines:**
        1.  **Formatting:** Do NOT use bold text or markdown formatting in your response. Use plain text only.
        2.  **For Definitions (e.g., "what is IPC"):**
            - Provide a clear, grounded explanation of the legal term or concept.
            - Follow up with a specific, practical example. For instance, after explaining the Indian Penal Code, you could mention, "A well-known example is Section 302 of the IPC, which deals with the punishment for murder."
        3.  **For Actionable Advice (e.g., "I am being cyberbullied"):**
            - Structure your response in three distinct parts:
            - **Your Rights:** Begin by explaining the user's legal rights under Indian law concerning their situation.
            - **What You Can Do:** Provide a clear, step-by-step list of actions the user can take.
            - **Legal Precedents:** If applicable, briefly mention relevant past cases or legal principles that support their rights, without going into excessive detail.

        **User Query:** "{query}"

        Please provide your answer below, following all guidelines.
        """
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"[ERROR][GEMINI] Gemini API call failed: {e}")
        return "I am sorry, but I was unable to find an answer in the provided documents or from my knowledge base."

# --- Upload Service ---
async def process_and_upload_document(file_path: str, licenseID: str) -> Dict[str, Any]:
    base_name = os.path.basename(file_path)
    print(f"[UPLOAD] Starting processing for: {base_name}")
    try:
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()

        pdf_file_obj = BytesIO(pdf_bytes)
        pdf = PyPDF2.PdfReader(pdf_file_obj)
        pages = [p.extract_text() or "" for p in pdf.pages]
        full_text = " ".join([t.replace("\n", " ") for t in pages]).strip()
        
        ocr_used = False
        if not full_text:
            print("[UPDATE] PDF text empty, using OCR fallback...")
            full_text = custom_update_helpers.return_string_from_path(pdf_bytes) or ""
            ocr_used = True

        if not full_text:
            raise ValueError("No text could be extracted from the PDF.")

        chunks = chunk_text(full_text)
        print(f"[UPLOAD] Document split into {len(chunks)} chunks.")

        upserts = []
        for idx, chunk in enumerate(chunks):
            vector_id = f"{licenseID}:{base_name}:{idx}"
            
            keywords_manual = custom_update_helpers.check_manual_keywords(chunk)
            keyword_corpus = custom_update_helpers.distill_string(chunk)
            key = custom_update_helpers.return_keyword(keyword_corpus, 30)
            keys = (keywords_manual + key)
            
            summary = make_summary(chunk, use_command_r=True)
            gpt_meta = get_title_date_parties(" ".join(chunk.split()[:300]))

            doc = {
                "vector_id": vector_id, "licenseID": licenseID, "file": base_name,
                "chunk_index": idx, "ocr": ocr_used, "keywords": [k.lower() for k in keys],
                "summary": summary, "text": chunk,
                "title": gpt_meta.get("title", ""), "parties": gpt_meta.get("parties", ""),
                "court": gpt_meta.get("court", ""), "date": gpt_meta.get("date", ""),
            }
            documents_collection.update_one({"vector_id": vector_id}, {"$set": doc}, upsert=True)
            
            emb = get_embedding(chunk)
            meta = sanitize_metadata({
                "licenseID": licenseID, "file": base_name, "chunk_index": idx,
                "title": doc["title"], "court": doc["court"], "date": doc["date"],
            })
            upserts.append({"id": vector_id, "values": emb, "metadata": meta})

        if upserts:
            pc_index.upsert(vectors=upserts, batch_size=100)
            print(f"[SUCCESS] Upserted {len(upserts)} vectors to Pinecone.")

        return {"file": base_name, "licenseID": licenseID, "chunks_processed": len(chunks)}
    
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# --- Search Service ---
async def perform_semantic_search(query: str, top_k: int, licenseID: str = None) -> Dict[str, Any]:
    print(f"[SEARCH] Received query: '{query}'")
    try:
        q_emb = get_embedding(query)
        query_params = {"vector": q_emb, "top_k": top_k, "include_metadata": True}
        if licenseID:
            query_params["filter"] = {"licenseID": licenseID}
        
        res = pc_index.query(**query_params)
        matches = res.get("matches", [])

        vids = [m["id"] for m in matches]
        mongo_docs = list(documents_collection.find({"vector_id": {"$in": vids}}))
        
        doc_map = {doc['vector_id']: doc for doc in mongo_docs}
        
        retrieved_docs_with_scores = []
        for m in matches:
            doc = doc_map.get(m['id'])
            if doc:
                doc['_id'] = str(doc['_id'])
                doc['retrieval_score'] = m.get('score', 0)
                retrieved_docs_with_scores.append(doc)
        
        if not retrieved_docs_with_scores:
            print("[INFO] No documents found. Using Gemini fallback.")
            fallback_answer = get_gemini_fallback_answer(query)
            return {"answer": fallback_answer, "docs": []}

        contexts = [doc['text'] for doc in retrieved_docs_with_scores]
        context_embeddings = reranking_model.encode(contexts, convert_to_tensor=True)
        query_embedding = reranking_model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, context_embeddings)[0].cpu().tolist()
        
        for doc, score in zip(retrieved_docs_with_scores, scores):
            doc['rerank_score'] = score
            
        ranked_docs = sorted(retrieved_docs_with_scores, key=lambda x: x['rerank_score'], reverse=True)

        context_blob = "\n\n---\n\n".join([doc['text'] for doc in ranked_docs])
        generated_answer = get_doc_based_judgement(query, context_blob)

        if not generated_answer.strip() or "does not provide information" in generated_answer.lower():
            print("[INFO] RAG answer was insufficient. Using Gemini fallback.")
            generated_answer = get_gemini_fallback_answer(query)
            
        return {"answer": generated_answer, "docs": ranked_docs}

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
