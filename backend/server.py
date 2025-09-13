from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os

from rag1 import ingest_file, ask_query

app = FastAPI()

# Enable CORS (adjust origins if you only want to allow your frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g. ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- MODELS ----------
class AskPayload(BaseModel):
    query: str


# ---------- ROUTES ----------
@app.post("/ingest")
async def ingest_endpoint(file: UploadFile, append: bool = Form(False)):
    os.makedirs("uploads", exist_ok=True)
    path = f"uploads/{file.filename}"

    with open(path, "wb") as f:
        f.write(await file.read())

    print(f"[INGEST] File saved: {file.filename}, append={append}", flush=True)

    try:
        ingest_file(path, append=append)
    except Exception as e:
        print(f"[ERROR][INGEST] {str(e)}", flush=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

    return {"status": "success", "file": file.filename, "append": append}


@app.post("/ask")
async def ask_endpoint(request: Request):
    raw_body = await request.body()
    print(f"[ASK][RAW BODY] {raw_body}", flush=True)

    try:
        data = await request.json()
        query = data.get("query") or data.get("search_key")  # accept either
        if not query or not query.strip():
            return JSONResponse(status_code=400, content={"error": "Missing 'query'"})

        print(f"[ASK] Received query: {query}", flush=True)
        answer = ask_query(query)
        print(f"[ASK] Sending response: {answer}", flush=True)
        return {"answer": answer}

    except Exception as e:
        print(f"[ERROR][ASK] {str(e)}", flush=True)
        return JSONResponse(status_code=500, content={"error": str(e)})



# ---------- MAIN ----------
if __name__ == "__main__":
    uvicorn.run(
        "rag_multi_doc_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug"
    )
