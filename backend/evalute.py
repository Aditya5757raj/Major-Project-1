#!/usr/bin/env python3
"""
evaluate_rag_lqrag_comparison.py
Enhanced evaluation script following LQ-RAG paper methodology (IEEE Access 2025)
Reference: DOI 10.1109/ACCESS.2025.3542125

Implements all key metrics from the paper:
- RAG Triad: Answer Relevance, Context Relevance, Groundedness
- Generation Quality: BLEU, ROUGE-L, Exact Match, Accuracy
- Retrieval Quality: Precision@K, Hit Rate, MRR
- Semantic Similarity
"""

import time
import numpy as np
import json
import pandas as pd
from typing import List, Dict, Tuple
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from rapidfuzz import fuzz

# Import your RAG pipeline functions
from rag1 import hybrid_retrieve, ask_query, embed_texts

# ==================== CONFIG ====================
JSON_FILE = "ipc_qa.json"
TOP_N = 10  # Changed from 10 to 50 for better statistical significance
TOP_K = 15  # Changed from 8 to 15 (paper uses k=15, Section VI-A, Figure 4)

# ==================== LQ-RAG PAPER BASELINE SCORES ====================
# These are the benchmark scores from the research paper to compare against
LQRAG_PAPER_SCORES = {
    "Naive RAG": {
        "answer_relevance": 0.87,      # Average from Table 11
        "context_relevance": 0.37,     # Average from Table 11
        "groundedness": 0.72,          # Average from Table 11
        "avg_response_time_ms": 7200,  # Figure 9
        "relevance_score": 0.65        # Figure 7
    },
    "RAG + FTM": {
        "answer_relevance": 0.92,      # Average from Table 11
        "context_relevance": 0.42,     # Average from Table 11
        "groundedness": 0.77,          # Average from Table 11
        "avg_response_time_ms": 11200, # Figure 9
        "relevance_score": 0.70        # Figure 7
    },
    "LQ-RAG": {
        "answer_relevance": 0.88,      # Average from Table 11
        "context_relevance": 0.70,     # Average from Table 11
        "groundedness": 0.82,          # Average from Table 11
        "avg_response_time_ms": 14600, # Figure 9
        "relevance_score": 0.80,       # Figure 7
        "hit_rate@5": 0.51,            # Table 6 (GIST-Law-Embed)
        "hit_rate@15": 0.63,           # Extrapolated from Figure 4
        "mrr": 0.40                    # Table 7 (GIST-Law-Embed)
    }
}

# Paper improvements (from Abstract and Section VI)
PAPER_IMPROVEMENTS = {
    "embedding_hit_rate": 0.13,     # 13% improvement
    "embedding_mrr": 0.15,          # 15% improvement
    "hfm_vs_general_llm": 0.24,     # 24% improvement
    "lqrag_vs_naive": 0.23,         # 23% improvement in relevance
    "lqrag_vs_ftm": 0.14            # 14% improvement over RAG+FTM
}

# ==================== METRICS IMPLEMENTATION ====================
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
smoothie = SmoothingFunction().method4

def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def tokenize(text: str) -> List[str]:
    """Simple tokenization"""
    return re.findall(r'\w+', text.lower())

# ---------- Retrieval Metrics (Section V-C.1, V-C.2) ----------

def hit_rate_at_k(retrieved_idxs: List[int], relevant_idxs: List[int], k: int) -> float:
    """
    Hit Rate @ K (Equation 5 in paper, page 7)
    Measures if correct answer is in top-k retrieved documents
    """
    retrieved_topk = set(retrieved_idxs[:k])
    relevant_set = set(relevant_idxs)
    return 1.0 if len(retrieved_topk & relevant_set) > 0 else 0.0

def mean_reciprocal_rank(retrieved_idxs: List[int], relevant_idxs: List[int]) -> float:
    """
    Mean Reciprocal Rank (Equation 6 in paper, page 7)
    Evaluates ranking quality
    """
    relevant_set = set(relevant_idxs)
    for rank, idx in enumerate(retrieved_idxs, start=1):
        if idx in relevant_set:
            return 1.0 / rank
    return 0.0

def precision_at_k(retrieved_idxs: List[int], relevant_idxs: List[int], k: int = 3) -> float:
    """Standard Precision@K"""
    retrieved_topk = retrieved_idxs[:k]
    relevant_set = set(relevant_idxs)
    correct = sum(1 for idx in retrieved_topk if idx in relevant_set)
    return correct / k if k > 0 else 0.0

# ---------- RAG Triad Metrics (Section V-C.4, V-C.5, V-C.6) ----------

def answer_relevance(question: str, answer: str, threshold: float = 0.6) -> float:
    """
    Answer Relevance (Equation 8 in paper, page 7)
    Measures how well answer addresses the query
    Note: Paper uses GPT-4 for scoring. This is simplified version.
    """
    q_tokens = set(tokenize(question))
    a_tokens = set(tokenize(answer))
    
    if len(q_tokens) == 0:
        return 0.0
    
    overlap = len(q_tokens & a_tokens)
    relevance = overlap / len(q_tokens)
    
    # Also check semantic completeness
    if len(answer.strip()) < 10:  # Too short answer
        relevance *= 0.5
    
    return min(1.0, relevance)

def context_relevance(question: str, context: str, threshold: float = 0.5) -> float:
    """
    Context Relevance (Equation 9 in paper, page 7)
    Measures how well retrieved context fits the query
    """
    q_tokens = set(tokenize(question))
    c_tokens = set(tokenize(context))
    
    if len(q_tokens) == 0:
        return 0.0
    
    overlap = len(q_tokens & c_tokens)
    relevance = overlap / len(q_tokens)
    
    return min(1.0, relevance)

def groundedness(answer: str, context: str) -> float:
    """
    Groundedness (Equation 10 in paper, page 7-8)
    Assesses if answer is grounded in retrieved context
    """
    a_tokens = set(tokenize(answer))
    c_tokens = set(tokenize(context))
    
    if len(a_tokens) == 0:
        return 0.0
    
    # Calculate what fraction of answer is supported by context
    overlap = len(a_tokens & c_tokens)
    ground_score = overlap / len(a_tokens)
    
    return min(1.0, ground_score)

# ---------- Generation Quality Metrics (Section V-C.7, V-C.8, V-C.9) ----------

def exact_match(pred: str, gold: str) -> float:
    """
    Exact Match (Equation 12 in paper, page 8)
    Strict string matching
    """
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0

def accuracy(pred: str, gold: str, threshold: float = 0.8) -> float:
    """
    Accuracy (Equation 11 in paper, page 8)
    Fuzzy matching for legal texts
    """
    similarity = fuzz.ratio(normalize_text(pred), normalize_text(gold)) / 100.0
    return 1.0 if similarity >= threshold else 0.0

def compute_bleu(pred: str, gold_list: List[str]) -> float:
    """
    BLEU Score (Equation 13 in paper, page 8)
    N-gram precision metric
    """
    references = [g.split() for g in gold_list]
    candidate = pred.split()
    if len(candidate) == 0:
        return 0.0
    return sentence_bleu(references, candidate, smoothing_function=smoothie)

def compute_rouge_scores(pred: str, gold: str) -> Dict[str, float]:
    """
    ROUGE Scores (Equation 14 in paper, page 8)
    Recall-oriented overlap metric
    """
    scores = scorer.score(gold, pred)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

def semantic_similarity(pred: str, gold: str) -> float:
    """
    Cosine Similarity (Equation 7 in paper, page 7)
    Semantic similarity using embeddings
    """
    try:
        pred_vec = embed_texts([pred])
        gold_vec = embed_texts([gold])
        
        if pred_vec.size == 0 or gold_vec.size == 0:
            return 0.0
        
        # Normalize vectors
        pred_norm = pred_vec / (np.linalg.norm(pred_vec) + 1e-8)
        gold_norm = gold_vec / (np.linalg.norm(gold_vec) + 1e-8)
        
        cos_sim = (pred_norm @ gold_norm.T)[0, 0]
        return float(np.clip(cos_sim, 0.0, 1.0))
    except Exception as e:
        print(f"Warning: Semantic similarity computation failed: {e}")
        return 0.0

# ---------- Automatic Relevant Chunk Detection ----------

def get_relevant_chunks(hits: List[Dict], gold_answer: str, topk: int = 3) -> List[int]:
    """
    Automatically determine relevant chunks using fuzzy matching
    Returns chunk indices that best match the gold answer
    """
    scores = []
    for h in hits:
        # Use multiple matching strategies
        partial_score = fuzz.partial_ratio(h['text'], gold_answer)
        token_score = fuzz.token_set_ratio(h['text'], gold_answer)
        combined_score = (partial_score + token_score) / 2
        scores.append((h['pos_global'], combined_score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, score in scores[:topk] if score > 40]  # threshold

# ==================== EVALUATION LOOP ====================

def evaluate_rag_system(dataset: List[Dict], top_n: int = 50, top_k: int = 15) -> Tuple[pd.DataFrame, Dict]:
    """
    Complete RAG evaluation following LQ-RAG paper methodology
    """
    results = []
    
    for idx, entry in enumerate(dataset[:top_n], start=1):
        question = entry['question']
        gold_answer = entry['answer']
        
        print(f"\n[{idx}/{min(top_n, len(dataset))}] Evaluating: {question[:80]}...")
        
        # ===== RETRIEVAL PHASE =====
        start_time = time.time()
        hits = hybrid_retrieve(question, k=top_k)
        retrieval_time = (time.time() - start_time) * 1000
        
        retrieved_ids = [h['pos_global'] for h in hits]
        retrieved_texts = [h['text'] for h in hits]
        combined_context = " ".join(retrieved_texts[:5])  # Top 5 chunks
        
        # ===== GENERATION PHASE =====
        gen_start = time.time()
        model_answer = ask_query(question)
        generation_time = (time.time() - gen_start) * 1000
        
        total_time = retrieval_time + generation_time
        
        # ===== DETERMINE RELEVANT CHUNKS =====
        relevant_chunk_ids = get_relevant_chunks(hits, gold_answer, topk=3)
        
        # ===== COMPUTE ALL METRICS =====
        
        # Retrieval Metrics (Table 6, 7)
        hr_3 = hit_rate_at_k(retrieved_ids, relevant_chunk_ids, k=3)
        hr_5 = hit_rate_at_k(retrieved_ids, relevant_chunk_ids, k=5)
        hr_10 = hit_rate_at_k(retrieved_ids, relevant_chunk_ids, k=10)
        mrr = mean_reciprocal_rank(retrieved_ids, relevant_chunk_ids)
        prec_3 = precision_at_k(retrieved_ids, relevant_chunk_ids, k=3)
        
        # RAG Triad Metrics (Figure 8, Table 11)
        ans_rel = answer_relevance(question, model_answer)
        ctx_rel = context_relevance(question, combined_context)
        ground = groundedness(model_answer, combined_context)
        
        # Generation Quality Metrics (Table 8, 9)
        em = exact_match(model_answer, gold_answer)
        acc = accuracy(model_answer, gold_answer)
        bleu = compute_bleu(model_answer, [gold_answer])
        rouge = compute_rouge_scores(model_answer, gold_answer)
        sem_sim = semantic_similarity(model_answer, gold_answer)
        
        # Overall Relevance Score (as per Figure 7)
        # Combined metric: weighted average of key metrics
        relevance_score = (ans_rel * 0.4 + ctx_rel * 0.3 + ground * 0.3)
        
        results.append({
            # Identifiers
            "question": question,
            "gold_answer": gold_answer,
            "model_answer": model_answer,
            
            # Retrieval Metrics
            "Hit_Rate@3": hr_3,
            "Hit_Rate@5": hr_5,
            "Hit_Rate@10": hr_10,
            "MRR": mrr,
            "Precision@3": prec_3,
            
            # RAG Triad Metrics (PRIMARY - Figure 8)
            "Answer_Relevance": ans_rel,
            "Context_Relevance": ctx_rel,
            "Groundedness": ground,
            "Overall_Relevance": relevance_score,
            
            # Generation Quality Metrics
            "Exact_Match": em,
            "Accuracy": acc,
            "BLEU": bleu,
            "ROUGE-1": rouge['rouge1'],
            "ROUGE-2": rouge['rouge2'],
            "ROUGE-L": rouge['rougeL'],
            "Semantic_Similarity": sem_sim,
            
            # Performance Metrics
            "Retrieval_Time_ms": retrieval_time,
            "Generation_Time_ms": generation_time,
            "Total_Response_Time_ms": total_time,
            
            # Retrieved chunks info
            "num_relevant_chunks_found": len(relevant_chunk_ids),
            "top_chunk_id": retrieved_ids[0] if retrieved_ids else -1
        })
        
        # Print progress
        if idx % 10 == 0:
            print(f"  Progress: {idx}/{min(top_n, len(dataset))} completed")
    
    df = pd.DataFrame(results)
    
    # Compute aggregate statistics
    aggregate_stats = {
        "total_questions": len(df),
        "avg_hit_rate@5": df["Hit_Rate@5"].mean(),
        "avg_mrr": df["MRR"].mean(),
        "avg_answer_relevance": df["Answer_Relevance"].mean(),
        "avg_context_relevance": df["Context_Relevance"].mean(),
        "avg_groundedness": df["Groundedness"].mean(),
        "avg_overall_relevance": df["Overall_Relevance"].mean(),
        "avg_exact_match": df["Exact_Match"].mean(),
        "avg_bleu": df["BLEU"].mean(),
        "avg_rouge_l": df["ROUGE-L"].mean(),
        "avg_semantic_similarity": df["Semantic_Similarity"].mean(),
        "avg_response_time_ms": df["Total_Response_Time_ms"].mean(),
    }
    
    return df, aggregate_stats

# ==================== COMPARISON WITH PAPER ====================

def compare_with_paper(your_stats: Dict) -> pd.DataFrame:
    """
    Compare your results with LQ-RAG paper benchmarks
    """
    comparison_data = []
    
    # Map your metrics to paper metrics
    metric_mapping = {
        "Answer Relevance": ("avg_answer_relevance", "answer_relevance"),
        "Context Relevance": ("avg_context_relevance", "context_relevance"),
        "Groundedness": ("avg_groundedness", "groundedness"),
        "Overall Relevance": ("avg_overall_relevance", "relevance_score"),
        "Hit Rate@5": ("avg_hit_rate@5", "hit_rate@5"),
        "MRR": ("avg_mrr", "mrr"),
        "Avg Response Time (ms)": ("avg_response_time_ms", "avg_response_time_ms")
    }
    
    for metric_name, (your_key, paper_key) in metric_mapping.items():
        your_score = your_stats.get(your_key, 0.0)
        
        row = {
            "Metric": metric_name,
            "Your Model": f"{your_score:.4f}",
        }
        
        # Add paper scores
        for model_name in ["Naive RAG", "RAG + FTM", "LQ-RAG"]:
            paper_score = LQRAG_PAPER_SCORES[model_name].get(paper_key, None)
            if paper_score is not None:
                row[f"Paper: {model_name}"] = f"{paper_score:.4f}"
                
                # Calculate improvement
                if paper_key != "avg_response_time_ms":
                    improvement = ((your_score - paper_score) / paper_score * 100) if paper_score > 0 else 0
                    row[f"Δ vs {model_name}"] = f"{improvement:+.1f}%"
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

# ==================== MAIN EXECUTION ====================

def main():
    print("="*80)
    print("RAG EVALUATION - LQ-RAG Paper Comparison")
    print("Paper Reference: Legal Query RAG (IEEE Access 2025)")
    print("DOI: 10.1109/ACCESS.2025.3542125")
    print("="*80)
    
    # Load dataset
    print(f"\nLoading dataset from {JSON_FILE}...")
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    print(f"Total questions in dataset: {len(dataset)}")
    print(f"Evaluating top {TOP_N} questions with k={TOP_K} retrieval")
    
    # Run evaluation
    print("\n" + "="*80)
    print("STARTING EVALUATION...")
    print("="*80)
    
    df, stats = evaluate_rag_system(dataset, top_n=TOP_N, top_k=TOP_K)
    
    # Save detailed results
    output_file = f"evaluation_lqrag_comparison_top{TOP_N}.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Detailed results saved to: {output_file}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("YOUR MODEL - AVERAGE METRICS")
    print("="*80)
    for metric, value in stats.items():
        print(f"{metric:30s}: {value:.4f}")
    
    # Compare with paper
    print("\n" + "="*80)
    print("COMPARISON WITH LQ-RAG PAPER BENCHMARKS")
    print("="*80)
    comparison_df = compare_with_paper(stats)
    print(comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_file = f"paper_comparison_top{TOP_N}.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\n✓ Comparison table saved to: {comparison_file}")
    
    # Print paper's key findings
    print("\n" + "="*80)
    print("LQ-RAG PAPER KEY FINDINGS (for reference)")
    print("="*80)
    print(f"• Embedding fine-tuning improved Hit Rate by {PAPER_IMPROVEMENTS['embedding_hit_rate']*100:.0f}%")
    print(f"• Embedding fine-tuning improved MRR by {PAPER_IMPROVEMENTS['embedding_mrr']*100:.0f}%")
    print(f"• HFM model showed {PAPER_IMPROVEMENTS['hfm_vs_general_llm']*100:.0f}% gain over general LLMs")
    print(f"• LQ-RAG achieved {PAPER_IMPROVEMENTS['lqrag_vs_naive']*100:.0f}% improvement over Naive RAG")
    print(f"• LQ-RAG achieved {PAPER_IMPROVEMENTS['lqrag_vs_ftm']*100:.0f}% improvement over RAG+FTM")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()