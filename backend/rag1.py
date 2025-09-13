#!/usr/bin/env python3
"""
rag_multi_doc.py
Generic Multi-Document RAG pipeline (FAISS + BM25 + hybrid retrieval + Gemini).

Usage:
  Ingest (create new index):
    python rag_multi_doc.py --ingest path/to/file.pdf

  Ingest and append to existing index:
    python rag_multi_doc.py --ingest path/to/file.pdf --append

  Ask query:
    python rag_multi_doc.py --ask "your question here"

Files produced:
  - legal_index.faiss  (FAISS index storing vectors)
  - meta.json          (list of metadata entries for ALL chunks)
  - doc_vecs.npy       (numpy array of all doc vectors in same order as meta.json)

NOT LEGAL ADVICE. Use responsibly.
"""
import os
import re
import json
import uuid
import argparse
import unicodedata
from typing import List, Dict, Any
from dotenv import load_dotenv   # NEW
load_dotenv()                    # Load .env before reading keys

import fitz                                    # pip install pymupdf
from unidecode import unidecode                # pip install unidecode
import numpy as np
import faiss                                   # pip install faiss-cpu
import tiktoken                                # pip install tiktoken
import google.generativeai as genai            # pip install google-generativeai
from rank_bm25 import BM25Okapi                # pip install rank_bm25
from rapidfuzz import fuzz                     # pip install rapidfuzz

# ---------- CONFIG ----------
EMBED_MODEL = os.getenv("GENAI_EMBED_MODEL", "text-embedding-004")
GEMINI_MODEL = os.getenv("GENAI_GEMINI_MODEL", "gemini-2.5-flash")
API_KEY = os.getenv("GOOGLE_API_KEY", None)
if API_KEY is None:
    raise RuntimeError("Set GOOGLE_API_KEY environment variable.")
genai.configure(api_key=API_KEY)

FAISS_PATH = "legal_index.faiss"
META_PATH = "meta.json"
DOC_VEC_PATH = "doc_vecs.npy"

MAX_TOK_PER_CHUNK = 520
OVERLAP_TOK = 160
TOP_K = 8

# ---------- Text extraction & normalization ----------
BAD_CHARS = r'[\u200B-\u200D\u2060\u00AD]'

def extract_pdf_text(path: str) -> str:
    doc = fitz.open(path)
    pages = [p.get_text("text") for p in doc]
    return "\n".join(pages)

def normalize_legal_text(s: str) -> str:
    s = re.sub(BAD_CHARS, '', s)
    s = s.replace('\r', '')
    s = re.sub(r'(\w)-\n(\w)', r'\1\2', s)
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    s = unicodedata.normalize('NFKC', s)
    s = unidecode(s)
    lines = s.split('\n')
    kept = []
    for ln in lines:
        if re.match(r'^\s*Page\s+\d+(\s+of\s+\d+)?\s*$', ln): continue
        if re.match(r'^\s*\d{1,3}\s*$', ln): continue
        kept.append(ln)
    return "\n".join(kept).strip()

# ---------- Token encoding and chunking ----------
ENC = tiktoken.get_encoding("cl100k_base")

def split_into_chunks(text: str, max_tok: int = MAX_TOK_PER_CHUNK, overlap: int = OVERLAP_TOK) -> List[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf, buf_tok = [], [], 0
    for p in paras:
        ptok = len(ENC.encode(p))
        if buf and buf_tok + ptok > max_tok:
            chunks.append("\n\n".join(buf))
            tail_tokens = ENC.encode("\n\n".join(buf))[-overlap:]
            tail = ENC.decode(tail_tokens) if tail_tokens else ""
            buf = [tail, p] if tail else [p]
            buf_tok = len(ENC.encode("\n\n".join(buf)))
        else:
            buf.append(p)
            buf_tok += ptok
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks

# ---------- Embeddings helpers (robust) ----------
def embed_texts(texts: List[str]) -> np.ndarray:
    """Call GenAI embed_content and return numpy array (n, dim) float32. Robust to response shapes."""
    if not texts:
        return np.zeros((0, 0), dtype='float32')
    try:
        resp = genai.embed_content(model=EMBED_MODEL, content=texts, task_type="retrieval_document")
    except Exception as e:
        raise RuntimeError(f"Embedding API error: {e}")

    embs = []
    if isinstance(resp, dict):
        if "data" in resp and isinstance(resp["data"], list):
            for item in resp["data"]:
                if isinstance(item, dict) and "embedding" in item:
                    embs.append(np.array(item["embedding"], dtype='float32'))
                elif isinstance(item, (list, tuple)):
                    embs.append(np.array(item, dtype='float32'))
        elif "embedding" in resp:
            emb_field = resp["embedding"]
            if isinstance(emb_field, list) and emb_field and isinstance(emb_field[0], (float, int)):
                embs.append(np.array(emb_field, dtype='float32'))
            elif isinstance(emb_field, list) and emb_field and isinstance(emb_field[0], dict):
                for itm in emb_field:
                    if "embedding" in itm:
                        embs.append(np.array(itm["embedding"], dtype='float32'))
            elif isinstance(emb_field, dict) and "embedding" in emb_field:
                embs.append(np.array(emb_field["embedding"], dtype='float32'))
            else:
                try:
                    for itm in emb_field:
                        embs.append(np.array(itm, dtype='float32'))
                except Exception:
                    raise RuntimeError("Could not parse embedding response format.")
        else:
            raise RuntimeError("Unexpected embedding response schema: keys=" + ",".join(resp.keys()))
    else:
        raise RuntimeError("Unexpected embedding response type: " + str(type(resp)))

    mat = np.vstack(embs).astype('float32')
    return mat

# ---------- Index & metadata persistence ----------
def create_index_from_vectors(vecs: np.ndarray, faiss_path: str = FAISS_PATH):
    if vecs.size == 0:
        raise RuntimeError("No vectors to build index.")
    faiss.normalize_L2(vecs)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    faiss.write_index(index, faiss_path)
    return index

def load_index(faiss_path: str = FAISS_PATH):
    if not os.path.exists(faiss_path):
        return None
    return faiss.read_index(faiss_path)

def save_meta(meta: List[Dict[str,Any]], meta_path: str = META_PATH):
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_meta(meta_path: str = META_PATH) -> List[Dict[str,Any]]:
    if not os.path.exists(meta_path):
        return []
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------- BM25 (improved tokenizer) ----------
STOP = set("the a an is are was were of to in on at for with by and or if then than from as be been being that this its".split())

def simple_tokens(s: str):
    s = s.lower()
    toks = re.findall(r"[a-z0-9]+", s)
    return [t for t in toks if t not in STOP]

def bm25_setup_from_meta(meta):
    corpus = [m["text"] for m in meta]
    tokenized = [simple_tokens(c) for c in corpus]
    bm25 = BM25Okapi(tokenized)
    return bm25, tokenized

# ---------- Retrieval helpers: dense ids, sparse ids, RRF, MMR ----------
def dense_search_ids(index, q: str, k=TOP_K*5):
    qv = embed_texts([q])
    faiss.normalize_L2(qv)
    k = min(k, index.ntotal) if hasattr(index, "ntotal") else k
    scores, ids = index.search(qv, k)
    ids = ids[0].tolist()
    return [int(i) for i in ids if i >= 0]

def bm25_search_ids(bm25, tokenized, q: str, k=TOP_K*5):
    scores = bm25.get_scores(simple_tokens(q))
    order = np.argsort(scores)[::-1][:k]
    return [int(i) for i in order]

def rrf_rank_lists(rank_lists: List[List[int]], k=200, c=60.0):
    scores = {}
    for ranks in rank_lists:
        for r, idx in enumerate(ranks):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (c + r + 1)
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in ordered][:k]

def mmr_select(candidates: List[int], doc_vecs: np.ndarray, qvec: np.ndarray, k=TOP_K, lambda_=0.7):
    if doc_vecs is None or qvec is None:
        return candidates[:k]
    selected = []
    cand_set = list(dict.fromkeys(candidates))
    qv = qvec.reshape(-1)
    while len(selected) < min(k, len(cand_set)):
        best = None; best_score = -1e9
        for idx in list(cand_set):
            sim_q = float(np.dot(doc_vecs[idx], qv))
            sim_sel = 0.0
            if selected:
                sim_sel = max(float(np.dot(doc_vecs[idx], doc_vecs[j])) for j in selected)
            score = lambda_ * sim_q - (1 - lambda_) * sim_sel
            if score > best_score:
                best_score = score; best = idx
        if best is None:
            break
        selected.append(best)
        cand_set.remove(best)
    return selected

# ---------- Metadata extraction helper ----------
def extract_metadata(meta, source_file="upload.pdf"):
    if not meta or not isinstance(meta, list):
        return {}
    text = " ".join(m.get("text", "") for m in meta)
    petitioners = None
    respondents = None
    sections = list(set(re.findall(r"section\s+\d+[A-Za-z]*\s*(?:ipc|crpc|act)?", text, re.IGNORECASE))) or None
    dates = list(set(re.findall(r"\b\d{1,2}[- /.](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[- /.]\d{2,4}\b", text, re.IGNORECASE) +
                     re.findall(r"\b\d{1,2}[- /.]\d{1,2}[- /.]\d{2,4}\b", text))) or None
    first_page = meta[0].get("text", "")[:1000] if meta else ""
    case_title_match = re.search(r"([A-Z][A-Za-z\s\.\-&]+)\s+vs\.?\s+([A-Z][A-Za-z\s\.\-&]+?)(?:\s+on\b|,|\n)", first_page, re.IGNORECASE)
    case_title = f"{case_title_match.group(1).strip()} vs. {case_title_match.group(2).strip()}" if case_title_match else None
    return {
        "case_title": case_title,
        "petitioners": petitioners,
        "respondents": respondents,
        "sections": sections,
        "dates": dates,
        "source_file": source_file,
    }

# ---------- Critical phrases for force-inclusion ----------
CRITICAL_PHRASES = [
    "section 94", "matriculation", "high school certificate", "date of birth",
    "ossification", "medical age", "bone age", "pocso", "prohibition of child marriage"
]

def force_include_indices(meta, max_extra=3):
    hits = []
    for i, m in enumerate(meta):
        t = m["text"].lower()
        score = sum(t.count(p) for p in CRITICAL_PHRASES)
        if score > 0:
            hits.append((i, score))
    hits.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in hits[:max_extra]]

# ---------- Query expansion ----------
def expand_query(q: str) -> List[str]:
    q = q.strip()
    expansions = [
        q,
        q + " age determination Section 94 JJ Act school certificate vs medical test",
        "priority of documents for age determination school matriculation certificate vs medical ossification test",
        "when can a court order ossification test if matriculation or birth certificate exists"
    ]
    seen = set()
    out = []
    for e in expansions:
        if e not in seen:
            seen.add(e); out.append(e)
    return out

# ---------- Gemini answer ----------
SAFETY_SYSTEM_PROMPT = (
    "You are a legal information assistant (not a lawyer)."
    "Base your answer strictly on the retrieved context. "
    "If the context includes principles or references implying an answer, apply them logically and explain. "
    "Include legal provisions (sections, acts) mentioned in context. "
    "If context is completely irrelevant, say 'insufficient context'. "
    "Jurisdiction: {jurisdiction}, Domain: {domain}."
)


def derive_flags_from_passages(passages: List[Dict[str,Any]]) -> Dict[str,bool]:
    txt = " ".join(p.get("text", "") for p in passages).lower()
    age_doc_priority = ("section 94" in txt and ("matriculation" in txt or "high school" in txt))
    return {"age_doc_priority": bool(age_doc_priority)}

def answer_with_gemini(query: str, passages: List[Dict[str,Any]], jurisdiction: str, domain: str, metadata: Dict[str,Any]=None) -> str:
    context_parts = []
    if metadata:
        context_parts.append(f"[Extracted Metadata]\n{json.dumps(metadata, indent=2)}")
    flags = derive_flags_from_passages(passages)
    if flags.get("age_doc_priority"):
        context_parts.append("[Derived Rule] When a High School/Matriculation certificate exists, medical age test should not be ordered (priority per Section 94 JJ Act).")
    for i, p in enumerate(passages):
        src = p.get("source", "unknown")
        docid = p.get("doc_id", "unknown")
        context_parts.append(f"[Doc {i+1}] (doc:{docid} src:{src})\n{p['text']}\n")
    context_blob = "\n\n---\n\n".join(context_parts)
    prompt = SAFETY_SYSTEM_PROMPT.format(jurisdiction=jurisdiction, domain=domain) + \
             f"\nContext:\n{context_blob}\n\nUser question: {query}\n\nAnswer:"
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", None) or (resp.get("output") if isinstance(resp, dict) else None) or str(resp)
        return text.strip()
    except Exception as e:
        return f"Gemini generation error: {e}"

# ---------- Ingest (supports append) ----------
def ingest_file(path: str, append: bool = False):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    raw = extract_pdf_text(path)
    clean = normalize_legal_text(raw)
    chunks = split_into_chunks(clean, MAX_TOK_PER_CHUNK, OVERLAP_TOK)

    # Build local meta entries for this doc
    doc_id = str(uuid.uuid4())
    source_name = os.path.basename(path)
    meta_entries = []
    for i, c in enumerate(chunks):
        meta_entries.append({
            "id": str(uuid.uuid4()),
            "doc_id": doc_id,
            "source": source_name,
            "text": c,
            "pos": i
        })

    print(f"[ingest] {len(chunks)} chunks from {source_name}")

    # compute embeddings for these chunks
    print("[ingest] creating embeddings for new chunks...")
    new_vecs = embed_texts([m["text"] for m in meta_entries])
    if new_vecs.size == 0:
        raise RuntimeError("Embedding API returned no vectors.")

    # Load existing meta/index if append
    if append and os.path.exists(META_PATH) and os.path.exists(FAISS_PATH) and os.path.exists(DOC_VEC_PATH):
        print("[ingest] appending to existing index.")
        # load existing meta and doc_vecs
        existing_meta = load_meta(META_PATH)
        existing_vecs = np.load(DOC_VEC_PATH)
        # concat meta
        combined_meta = existing_meta + meta_entries
        # concat vecs
        combined_vecs = np.vstack([existing_vecs, new_vecs]).astype('float32')
        # rebuild index from combined_vecs (safer than trying to mutate a saved flat index)
        print("[ingest] rebuilding FAISS index with combined vectors...")
        create_index_from_vectors(combined_vecs, FAISS_PATH)
        # persist
        np.save(DOC_VEC_PATH, combined_vecs)
        save_meta(combined_meta, META_PATH)
        print(f"[ingest] appended {len(meta_entries)} chunks. Total chunks: {len(combined_meta)}")
    else:
        print("[ingest] creating new index (overwrite mode).")
        all_meta = meta_entries
        vecs = new_vecs.astype('float32')
        create_index_from_vectors(vecs, FAISS_PATH)
        np.save(DOC_VEC_PATH, vecs)
        save_meta(all_meta, META_PATH)
        print(f"[ingest] created new index with {len(all_meta)} chunks.")

# ---------- Hybrid retrieve across all docs ----------
def hybrid_retrieve(q: str, k=TOP_K):
    if not os.path.exists(FAISS_PATH) or not os.path.exists(META_PATH) or not os.path.exists(DOC_VEC_PATH):
        raise FileNotFoundError("Index, meta, or doc_vecs missing. Run ingestion first.")

    index = load_index(FAISS_PATH)
    meta = load_meta(META_PATH)
    doc_vecs = np.load(DOC_VEC_PATH)

    bm25, tokenized = bm25_setup_from_meta(meta)

    # Expand queries, get dense and sparse id lists
    expanded = expand_query(q)
    dense_lists = []
    sparse_lists = []
    for eq in expanded:
        try:
            dense_ids = dense_search_ids(index, eq, k=TOP_K*5)
        except Exception:
            dense_ids = []
        sparse_ids = bm25_search_ids(bm25, tokenized, eq, k=TOP_K*5)
        dense_lists.append(dense_ids)
        sparse_lists.append(sparse_ids)

    fused = rrf_rank_lists(dense_lists + sparse_lists, k=200)

    # Force include critical indices
    forced = force_include_indices(meta, max_extra=3)
    for idx in reversed(forced):  # reversed to keep their order at start
        if idx in fused:
            fused.remove(idx)
        fused.insert(0, idx)

    # MMR selection using stored doc_vecs
    try:
        qv = embed_texts([q]).astype('float32'); faiss.normalize_L2(qv); qv = qv.reshape(-1)
        chosen_idxs = mmr_select(fused, doc_vecs, qv, k=k, lambda_=0.7)
    except Exception:
        chosen_idxs = fused[:k]

    results = []
    for idx in chosen_idxs:
        if 0 <= idx < len(meta):
            item = meta[idx].copy()
            item['pos_global'] = idx
            results.append(item)
    return results

# ---------- Ask query (assemble contexts + call Gemini) ----------
def ask_query(q: str):
    hits = hybrid_retrieve(q, k=TOP_K)
    # Build extracted metadata (basic)
    extracted_meta = extract_metadata(hits or [], source_file="multi-doc-collection")
    # Add a structured metadata doc for model visibility
    meta_blob = json.dumps(extracted_meta, indent=2)
    combined_contexts = hits.copy()
    combined_contexts.append({"id": "system_meta", "doc_id": "system", "source": "system", "text": f"DOCUMENT METADATA:\n{meta_blob}", "pos": -1})

    jur, dom = ("IN", "GENERIC")
    answer = answer_with_gemini(q, combined_contexts, jur, dom, metadata=extracted_meta)
    return {
        "answer": answer,
        "contexts": [
            {"doc_id": h.get("doc_id"), "source": h.get("source"), "pos_global": h.get("pos_global"), "snippet": h["text"][:800]}
            for h in hits
        ],
        "doc_metadata": extracted_meta
    }

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(description="Generic Multi-Doc RAG (FAISS + BM25 + Gemini)")
    p.add_argument("--ingest", help="Path to PDF to ingest")
    p.add_argument("--append", action="store_true", help="Append to existing index (use with --ingest)")
    p.add_argument("--ask", help="Query to ask against index")
    args = p.parse_args()

    if args.ingest:
        ingest_file(args.ingest, append=args.append)
    elif args.ask:
        out = ask_query(args.ask)
        print("\n--- ANSWER ---\n")
        print(out["answer"])
        print("\n--- DOCUMENT METADATA (structured) ---\n")
        print(json.dumps(out["doc_metadata"], indent=2))
        print("\n--- TOP CONTEXTS ---\n")
        for c in out["contexts"]:
            print(f"DOC:{c['doc_id']} SRC:{c['source']} POS:{c['pos_global']}\n{c['snippet']}\n---\n")
    else:
        p.print_help()

if __name__ == "__main__":
    main()
