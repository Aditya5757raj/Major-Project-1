import os
import json
import re
import logging
from typing import Dict, Any, List, Optional
import requests
from datetime import datetime
import backoff  # For retry mechanism
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
# --- Groq API Configuration ---
# Fetch the API key securely from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    # Fail fast if the API key is not configured
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")
# Groq API configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


# Model configuration
DEFAULT_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "gemma-2-9b-it"

# Rate limiting and retry configuration
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30  # seconds

def groq_generate(
    prompt: str, 
    temperature: float = 0.7, 
    max_tokens: int = 1000,
    model: str = DEFAULT_MODEL,
    system_message: Optional[str] = None
) -> str:
    """
    Send prompt to Groq API with retry mechanism and better error handling.
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 0.9,
    }

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, requests.exceptions.Timeout),
        max_tries=MAX_RETRIES,
        giveup=lambda e: isinstance(e, requests.exceptions.HTTPError) and e.response.status_code in [400, 401, 403, 404]
    )
    def make_request():
        try:
            logger.info(f"Sending request to Groq API with model: {model}")
            response = requests.post(
                GROQ_API_URL, 
                headers=headers, 
                json=payload,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            
            if not content:
                raise ValueError("Empty response from Groq API")
                
            return content
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.warning("Rate limit exceeded, will retry...")
                raise  # This will trigger retry
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            return f"Error: HTTP {e.response.status_code}"
        except requests.exceptions.Timeout:
            logger.warning("Request timeout, will retry...")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return f"Error generating response: {str(e)}"

    try:
        return make_request()
    except Exception as e:
        # Fallback to different model if primary fails
        if model != FALLBACK_MODEL:
            logger.warning(f"Primary model failed, trying fallback: {FALLBACK_MODEL}")
            return groq_generate(prompt, temperature, max_tokens, FALLBACK_MODEL, system_message)
        return f"Error: Failed after multiple retries and fallback: {str(e)}"

def get_judgement(search_text: str, context: Optional[str] = None) -> str:
    """
    Generate a judgment-like response based on Indian Constitution with optional context.
    """
    system_message = (
        "You are a legal expert specializing in Indian constitutional law. "
        "Provide precise, authoritative judgment analysis citing relevant articles and sections. "
        "Maintain professional, neutral language and structure your response like a legal judgment."
    )
    
    prompt = (
        "Provide a comprehensive legal judgment analysis based on the Indian Constitution. "
        "Include relevant articles, sections, and precedents where appropriate.\n\n"
    )
    
    if context:
        prompt += f"CONTEXT:\n{context}\n\n"
    
    prompt += (
        f"QUERY: {search_text}\n\n"
        "STRUCTURE YOUR RESPONSE AS:\n"
        "1. Brief overview of the legal issue\n"
        "2. Relevant constitutional provisions\n"
        "3. Analysis and reasoning\n"
        "4. Conclusion\n\n"
        "JUDGMENT ANALYSIS:"
    )
    
    return groq_generate(prompt, temperature=0.5, max_tokens=1500, system_message=system_message)

def get_title_date_parties(doc_text: str, max_words: int = 1000) -> Dict[str, Any]:
    """
    Extract metadata (title, date, parties, court) from a legal document with improved parsing.
    """
    words = doc_text.split()[:max_words]
    chunk = " ".join(words)
    
    system_message = (
        "You are a legal document parser. Extract metadata accurately and return ONLY valid JSON. "
        "For dates, use YYYY-MM-DD format. For parties, use 'Appellant vs Respondent' format."
    )
    
    prompt = (
        "Extract the following metadata from the legal document text:\n"
        "- title: The case title (string)\n"
        "- date: The judgment date in YYYY-MM-DD format (string or null)\n"
        "- parties: The involved parties in 'Appellant vs Respondent' format (string)\n"
        "- court: The court name (string)\n"
        "- case_number: The case number if available (string or null)\n\n"
        "Return ONLY valid JSON with exactly these keys. If any field cannot be determined, use null.\n\n"
        "DOCUMENT TEXT:\n{chunk}\n\n"
        "JSON RESPONSE:"
    ).format(chunk=chunk)
    
    response = groq_generate(prompt, temperature=0.1, max_tokens=500, system_message=system_message)
    
    # Improved JSON extraction
    json_match = re.search(r'\{[\s\S]*\}', response)
    if json_match:
        try:
            result = json.loads(json_match.group(0))
            # Validate required fields
            required_fields = ["title", "date", "parties", "court", "case_number"]
            for field in required_fields:
                if field not in result:
                    result[field] = None
            return result
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return {field: None for field in ["title", "date", "parties", "court", "case_number", "error"]}
    
    return {field: None for field in ["title", "date", "parties", "court", "case_number", "error"]}

def get_doc_based_judgement(
    query: str, 
    doc_text: str, 
    max_tokens: int = 3000,
    include_citations: bool = True
) -> str:
    """
    Answer a user's legal query strictly based on the given document with improved context handling.
    """
    if not doc_text.strip():
        return "No document content available to answer the query."

    # Smart truncation to preserve important sections
    if len(doc_text) > max_tokens:
        # Try to keep conclusion and key sections
        important_sections = ["HELD", "JUDGMENT", "CONCLUSION", "ORDER"]
        truncated = False
        
        for section in important_sections:
            section_match = re.search(fr"(?i){section}.*?$", doc_text, re.DOTALL | re.MULTILINE)
            if section_match:
                # Keep the important section and some context
                start_pos = max(0, section_match.start() - 500)
                doc_text = doc_text[start_pos:start_pos + max_tokens] + "...[document truncated but key sections preserved]"
                truncated = True
                break
        
        if not truncated:
            doc_text = doc_text[:max_tokens] + "...[document truncated]"

    system_message = (
        "You are a legal research assistant. Answer queries strictly based on the provided document. "
        "Do not use external knowledge. Be precise and cite specific parts of the document when possible."
    )
    
    prompt = (
        "STRICT INSTRUCTION: Answer the user's legal query using ONLY the information "
        "provided in the document text below. Do not use any external knowledge or make assumptions.\n\n"
        "If the document does not contain information to answer the query, respond with: "
        "'The document does not provide information to answer this query.'\n\n"
        "DOCUMENT TEXT:\n"
        f"{doc_text}\n\n"
        "USER QUERY:\n"
        f"{query}\n\n"
        "ANSWER (based strictly on document above):"
    )
    
    if include_citations:
        prompt += "\n\nWhere possible, cite specific sections or paragraphs from the document."

    return groq_generate(
        prompt, 
        temperature=0.1, 
        max_tokens=min(2000, max_tokens // 2),
        system_message=system_message
    )

def batch_process_queries(queries: List[str], doc_text: str) -> Dict[str, str]:
    """
    Process multiple queries against the same document efficiently.
    """
    results = {}
    for query in queries:
        results[query] = get_doc_based_judgement(query, doc_text)
    return results

# Utility functions
def validate_groq_api_key() -> bool:
    """Validate the Groq API key by making a test request."""
    try:
        test_response = groq_generate("Test", max_tokens=10)
        return not test_response.startswith("Error")
    except:
        return False

def get_usage_stats() -> Dict[str, Any]:
    """Get API usage statistics (placeholder for actual implementation)."""
    return {
        "last_request": datetime.now().isoformat(),
        "model_used": DEFAULT_MODEL,
        "status": "active"
    }

