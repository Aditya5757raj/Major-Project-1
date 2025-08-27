# main.py
import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware # Import the CORS middleware
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

# Import the core service functions
from services import process_and_upload_document, perform_semantic_search

# Create the FastAPI app instance
app = FastAPI(
    title="Legal Document AI API",
    description="API for uploading and semantically searching legal documents.",
    version="1.0.0"
)

# --- Add CORS Middleware ---
# This is the new section that fixes the error.
# It allows your frontend (running on any origin "*") to communicate with the backend.
origins = ["*"] # For development, allow all origins. For production, restrict this.

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"], # Allow all headers
)

# Define the directory for temporary file storage
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# --- Pydantic Models for Request Bodies ---
class SearchQuery(BaseModel):
    search_key: str = Field(..., description="The question or query string to search for.")
    top: int = Field(5, gt=0, le=50, description="The number of top results to retrieve.")
    licenseID: Optional[str] = Field(None, description="Filter results by a specific license ID. If null, searches globally.")

# --- API Endpoints ---

@app.get("/", tags=["Health Check"])
async def read_root():
    """A simple health check endpoint."""
    return {"status": "ok", "message": "Welcome to the Legal Document AI API"}

@app.post("/upload/", tags=["Documents"])
async def upload_document(
    licenseID: str = Form(...), 
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Uploads a PDF document, processes it, and stores it in the vector database.
    
    - **licenseID**: The unique identifier for the client/license.
    - **file**: The PDF file to be uploaded.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are accepted.")
    
    temp_file_path = os.path.join(TEMP_DIR, file.filename)
    
    try:
        # Save the uploaded file temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the document using the service function
        result = await process_and_upload_document(temp_file_path, licenseID)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {"status": "success", "data": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        # Ensure the temp file is cleaned up even if an error occurs
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/search/", tags=["Search"])
async def search_documents(query: SearchQuery = Body(...)) -> Dict[str, Any]:
    """
    Performs a semantic search across the indexed documents and returns only the generated answer.
    
    - **search_key**: The natural language query.
    - **top**: The number of documents to return.
    - **licenseID**: (Optional) Scopes the search to a specific license.
    """
    try:
        result = await perform_semantic_search(
            query=query.search_key,
            top_k=5,
            licenseID=query.licenseID
        )
        if "error" in result:
             raise HTTPException(status_code=500, detail=result["error"])
        
        # Only return the generated answer (judgement) and not the source documents
        final_response_data = {"answer": result.get("answer", "No answer could be generated.")}
        
        return {"status": "success", "data": final_response_data}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during search: {str(e)}")

# --- How to run the server ---
# Use the command: uvicorn main:app --reload
# Access the interactive API docs at http://127.0.0.1:8000/docs
