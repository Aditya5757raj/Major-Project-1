## Imports
from io import BytesIO
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
from yake import KeywordExtractor
import re
import json
import os
import spacy
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from wordfreq import zipf_frequency

## Loading stopwords
print("[INFO] Loading stopwords from stopwords.json...")
with open("stopwords.json", "r") as f:
    final_stop = set(json.load(f)['stopwords'])
print(f"[INFO] Loaded {len(final_stop)} stopwords.")

## Load SpaCy
print("[INFO] Loading SpaCy model: en_core_web_sm")
nlp = spacy.load("en_core_web_sm")
print("[INFO] SpaCy model loaded.")

# -------------------------------
# OCR Module
# -------------------------------
def return_string_from_path(file_bytes):
    print("[INFO] Converting PDF to images for OCR...")
    images = convert_from_bytes(file_bytes, size=800)
    print(f"[INFO] Converted PDF into {len(images)} pages.")
    pages_text = []

    for i, img in enumerate(images):
        print(f"[INFO] Running OCR on page {i+1}...")
        text_page = pytesseract.image_to_string(img, lang='eng')
        pages_text.append(text_page)
    combined_text = " ".join(pages_text).strip()
    print(f"[SUCCESS] OCR complete. Extracted {len(combined_text)} characters.")
    return combined_text

# -------------------------------
# Spell Correction
# -------------------------------
def spell_check(text, max_chars=2000):
    if len(text) > max_chars:
        print("[INFO] Text too long for full spell check; splitting...")
        chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
        corrected_chunks = [str(TextBlob(c).correct()) for c in chunks]
        corrected_text = " ".join(corrected_chunks)
    else:
        corrected_text = str(TextBlob(text).correct())
    return corrected_text.strip()

# -------------------------------
# Manual Keyword Extraction
# -------------------------------
def check_manual_keywords(text):
    text = text.lower()
    tokens = word_tokenize(text)
    keywords_detected = []
    pattern = re.compile(r'^(section|sec|article|clause)[\s\-]?(\d+[a-zA-Z]*)$')

    for i in range(len(tokens)-1):
        combined = tokens[i] + "-" + tokens[i+1]
        if pattern.match(tokens[i] + " " + tokens[i+1]) and combined not in keywords_detected:
            keywords_detected.append(combined)
            print(f"[FOUND] Manual keyword detected: {combined}")
    return keywords_detected

# -------------------------------
# Preprocessing Module
# -------------------------------
def distill_string(text):
    print("[INFO] Starting preprocessing...")
    original_len = len(text)
    text = text.lower()
    text = re.sub(r'(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?', '', text)
    text = re.sub(' +', ' ', text)
    
    # Remove stopwords
    tokens = [tok for tok in word_tokenize(text) if tok not in final_stop]
    text = " ".join(tokens)
    
    # Lemmatization
    lemmas = [token.lemma_ for token in nlp(text)]
    text = " ".join(lemmas)

    # Keep common English words only
    tokens_en = [w for w in word_tokenize(text) if zipf_frequency(w, 'en', wordlist='best') > 3.3]
    text = " ".join(tokens_en)

    # Optional spell correction
    text = spell_check(text)
    print(f"[SUCCESS] Preprocessing complete. Original length: {original_len}, Final length: {len(text)}")
    return text

# -------------------------------
# Keyword Extraction
# -------------------------------
def return_keyword(text, top_n=30):
    print(f"[INFO] Extracting top {top_n} keywords using YAKE...")
    kw_extractor = KeywordExtractor(lan="en", n=1, top=top_n)
    keywords = [kw[0] for kw in kw_extractor.extract_keywords(text)]
    print(f"[SUCCESS] Extracted {len(keywords)} keywords: {keywords}")
    return keywords
