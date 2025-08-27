import cohere
import time
import os
from utils import message
from dotenv import load_dotenv
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co_client = cohere.Client(COHERE_API_KEY)

# Constants
MAX_TOKENS_PER_CHUNK = 3000  # token limit per chunk
RATE_LIMIT = 5  # requests per minute
SLEEP_TIME = 60 / RATE_LIMIT  # seconds between requests


def chunk_text_tokenwise(text, max_tokens=MAX_TOKENS_PER_CHUNK):
    """
    Split text into chunks based on Cohere token count.
    """
    tokens = co_client.tokenize(text).tokens
    chunks = []
    current_chunk_tokens = []

    for token in tokens:
        current_chunk_tokens.append(token)
        if len(current_chunk_tokens) >= max_tokens:
            chunk_text = co_client.detokenize(current_chunk_tokens)
            chunks.append(chunk_text)
            current_chunk_tokens = []

    if current_chunk_tokens:
        chunk_text = co_client.detokenize(current_chunk_tokens)
        chunks.append(chunk_text)

    return chunks


def summarize_chunk(chunk, use_command_r):
    """
    Summarize a single chunk with legal context.
    """
    if use_command_r:
        prompt = f"""
You are an expert legal analyst specializing in Indian law. Create a comprehensive summary of the following legal text.

**REQUIRED ELEMENTS:**
1. Case title and citation
2. Parties involved (Appellant vs Respondent)
3. Court and bench details
4. Date of judgment
5. Relevant IPC sections, Articles, or legal provisions
6. Key facts and background
7. Legal issues framed
8. Court's reasoning and analysis
9. Final decision and order
10. Precedents cited (if any)
Document:
{chunk}
"""
        response = co_client.generate(model="command-r-plus", prompt=prompt)
        return response.generations[0].text.strip()
    else:
        # fallback to summarize-xlarge
        response = co_client.summarize(
            text=chunk,
            model="summarize-xlarge",
            length="medium",
            extractiveness="medium",
            format="paragraph",
        )
        return response.summary


def make_summary(text: str, use_command_r: bool = True):
    """
    Summarize legal text using Cohere.
    Only chunk if an exception occurs due to token limit.
    """
    try:
        summary = summarize_chunk(text, use_command_r)
        print(f"[INFO] Summary generated (length={len(summary)} chars)")
        return summary

    except Exception as e:
        error_msg = str(e)
        if "maximum context length" in error_msg.lower() or "token" in error_msg.lower():
            print("[WARNING] Token limit exceeded. Splitting text into token-accurate chunks.")
            chunks = chunk_text_tokenwise(text)
            final_summary = []

            for i, chunk in enumerate(chunks):
                print(f"[INFO] Summarizing chunk {i + 1}/{len(chunks)}")
                try:
                    chunk_summary = summarize_chunk(chunk, use_command_r)
                    final_summary.append(chunk_summary)
                    time.sleep(SLEEP_TIME)  # rate limiting
                except Exception as inner_e:
                    print(f"[ERROR] Failed to summarize chunk {i + 1}: {inner_e}")
                    final_summary.append(chunk)  # fallback to original text

            combined_summary = " ".join(final_summary)
            print(f"[INFO] Combined summary length: {len(combined_summary)} chars")
            return combined_summary
        else:
            print(f"[ERROR] Error in make_summary: {error_msg}")
            return message.message_error(500, error_msg, "Internal Server Error")
