# db_config.py
import os
import json
from dotenv import load_dotenv

import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from pymongo import MongoClient
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

print("[CONFIG] Loading configurations and models...")

# --- API Keys and Constants ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "legal-docs"
MONGO_URI = os.getenv("MONGO_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
INLEGAL_MODEL = "law-ai/InLegalBERT"
DB_NAME = "legal_docs"
COLLECTION_NAME = "documents"

# --- Configure Gemini API ---
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# --- MongoDB Connection ---
try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    documents_collection = db[COLLECTION_NAME]
    print("[SUCCESS] MongoDB connected.")
except Exception as e:
    print(f"[ERROR] MongoDB connection failed: {e}")
    raise

# --- Stopwords ---
try:
    with open("stopwords.json", "r") as f:
        final_stop = json.load(f)['stopwords']
    print("[SUCCESS] Stopwords loaded.")
except FileNotFoundError:
    print("[WARNING] stopwords.json not found.")
    final_stop = []

# --- Embedding Model (InLegalBERT) ---
print("[MODEL LOAD] Loading InLegalBERT for embeddings...")
embedding_tokenizer = AutoTokenizer.from_pretrained(INLEGAL_MODEL)
embedding_model = AutoModel.from_pretrained(INLEGAL_MODEL)
embedding_model.eval()
EMB_DIM = embedding_model.config.hidden_size

# --- Reranking Model ---
print("[MODEL LOAD] Loading all-MiniLM-L6-v2 for reranking...")
reranking_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# --- Pinecone v3 ---
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"[PINECONE] Creating index {PINECONE_INDEX_NAME} with dimension {EMB_DIM}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMB_DIM,
            metric="cosine"
        )
    pc_index = pc.Index(PINECONE_INDEX_NAME)
    print("[SUCCESS] Pinecone initialized.")
except Exception as e:
    print(f"[ERROR] Pinecone initialization failed: {e}")
    raise

print("\n[READY] All configurations and models loaded successfully! ✅\n")