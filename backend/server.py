from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from collections import deque
import uvicorn
import os
import mysql.connector
from mysql.connector import Error
from google import genai
from datetime import datetime

from rag1 import ingest_file

# ------------------- FastAPI App -------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- Gemini Setup -------------------
GEMINI_MODEL = os.getenv("GENAI_GEMINI_MODEL", "gemini-2.5-flash")
API_KEY = os.getenv("GOOGLE_API_KEY", None)
if API_KEY is None:
    raise RuntimeError("Set GOOGLE_API_KEY environment variable.")

client = genai.Client()

def ask_gemini_api(prompt: str) -> str:
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return response.text if hasattr(response, "text") else "No response from Gemini"
    except Exception as e:
        print(f"[GEMINI ERROR] {e}", flush=True)
        return "Error generating response from Gemini"

# ------------------- MySQL Setup -------------------
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="aditya2004",
        database="chatbot"
    )

# ------------------- Conversation Context -------------------
conversation_contexts = {}

# ------------------- ROUTES -------------------

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
    try:
        data = await request.json()
        query = data.get("query") or data.get("search_key")
        user_email = data.get("chat", {}).get("email")
        chat_id = data.get("chat_id")  # Get existing chat_id if provided

        print(f"[ASK] Query: {query}, User: {user_email}, Chat ID: {chat_id}", flush=True)

        if not query or not query.strip():
            return JSONResponse(status_code=400, content={"error": "Missing 'query'"})
        if not user_email:
            return JSONResponse(status_code=400, content={"error": "Missing user email"})

        # ------------------- Manage Conversation Context -------------------
        context_key = f"{user_email}_{chat_id}" if chat_id else user_email
        if context_key not in conversation_contexts:
            conversation_contexts[context_key] = deque(maxlen=5)
        context = conversation_contexts[context_key]

        context.append({"role": "user", "message": query})
        prompt = "\n".join([f"{c['role']}: {c['message']}" for c in context])
        answer = ask_gemini_api(prompt)
        context.append({"role": "assistant", "message": answer})

        # ------------------- Store in MySQL -------------------
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # If no chat_id, create new chat
            if not chat_id:
                # Create new chat with title from first query
                chat_title = query[:50] + "..." if len(query) > 50 else query
                cursor.execute(
                    "INSERT INTO chats (user_email, title, created_at) VALUES (%s, %s, %s)",
                    (user_email, chat_title, datetime.now())
                )
                chat_id = cursor.lastrowid
                print(f"[ASK] Created new chat with ID: {chat_id}", flush=True)
            
            # Insert message into messages table
            cursor.execute(
                "INSERT INTO messages (chat_id, role, message, created_at) VALUES (%s, %s, %s, %s)",
                (chat_id, "user", query, datetime.now())
            )
            cursor.execute(
                "INSERT INTO messages (chat_id, role, message, created_at) VALUES (%s, %s, %s, %s)",
                (chat_id, "assistant", answer, datetime.now())
            )
            
            # Update chat's updated_at timestamp
            cursor.execute(
                "UPDATE chats SET updated_at = %s WHERE id = %s",
                (datetime.now(), chat_id)
            )
            
            conn.commit()
            cursor.close()
            conn.close()
        except Error as db_err:
            print(f"[DB ERROR] {db_err}", flush=True)

        return {"answer": answer, "chat_id": chat_id, "user": user_email}

    except Exception as e:
        print(f"[ERROR][ASK] {str(e)}", flush=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/chats/{user_email}")
async def get_chats(user_email: str):
    """Get all chats for a user"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT id, title, created_at, updated_at FROM chats WHERE user_email=%s ORDER BY updated_at DESC",
            (user_email,)
        )
        chats = cursor.fetchall()
        cursor.close()
        conn.close()
        
        print(f"[CHATS] Found {len(chats)} chats for {user_email}", flush=True)
        return {"chats": chats, "user": user_email}
    except Error as db_err:
        print(f"[DB ERROR] {db_err}", flush=True)
        return JSONResponse(status_code=500, content={"error": str(db_err)})


@app.get("/messages/{chat_id}")
async def get_chat_messages(chat_id: int):
    """Get all messages for a specific chat"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT role, message, created_at FROM messages WHERE chat_id=%s ORDER BY created_at ASC",
            (chat_id,)
        )
        messages = cursor.fetchall()
        cursor.close()
        conn.close()
        
        print(f"[MESSAGES] Found {len(messages)} messages for chat {chat_id}", flush=True)
        return {"messages": messages}
    except Error as db_err:
        print(f"[DB ERROR] {db_err}", flush=True)
        return JSONResponse(status_code=500, content={"error": str(db_err)})


@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: int):
    """Delete a chat and all its messages"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Delete messages first (foreign key constraint)
        cursor.execute("DELETE FROM messages WHERE chat_id=%s", (chat_id,))
        # Delete chat
        cursor.execute("DELETE FROM chats WHERE id=%s", (chat_id,))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"[DELETE] Deleted chat {chat_id}", flush=True)
        return {"status": "success", "deleted_chat_id": chat_id}
    except Error as db_err:
        print(f"[DB ERROR] {db_err}", flush=True)
        return JSONResponse(status_code=500, content={"error": str(db_err)})


# ------------------- MAIN -------------------
if __name__ == "__main__":
    uvicorn.run(
        "rag_multi_doc_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug"
    )