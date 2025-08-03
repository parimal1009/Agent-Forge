import os
import uuid
import sqlite3
import requests
import tempfile
import re
import io
import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import PyPDF2
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import google.auth.transport.requests
import cachetools
import html

# --- Pydantic Models ---
class Agent(BaseModel):
    id: str
    name: str
    model: str
    description: Optional[str] = None
    created_at: str
    performance_score: float
    total_conversations: int
    embed_enabled: bool

class AgentUpdate(BaseModel):
    name: Optional[str] = None
    model: Optional[str] = None
    description: Optional[str] = None
    embed_enabled: Optional[bool] = None

class ConversationCreate(BaseModel):
    message: str
    session_id: str

class ConversationResponse(BaseModel):
    response: str
    agent_id: str
    session_id: str

# --- Configuration ---
app = FastAPI(title="AgentForge", version="1.0.0", description="AI Agent Management System")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
os.makedirs("vectorstores", exist_ok=True)
os.makedirs("google_tokens", exist_ok=True)

# Google Drive configuration
GOOGLE_SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
GROQ_MODELS = [
    "gemma2-9b-it",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "whisper-large-v3",
    "whisper-large-v3-turbo",
    "distil-whisper-large-v3-en",
    "deepseek-r1-distill-llama-70b",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama3-groq-70b-8192-tool-use-preview",
    "llama3-groq-8b-8192-tool-use-preview"
]

# --- Caching ---
vectorstore_cache = cachetools.TTLCache(maxsize=100, ttl=3600)  # 1 hour cache
chain_cache = cachetools.TTLCache(maxsize=100, ttl=1800)  # 30 minute cache

# --- Rate Limiting ---
rate_limits = {}
RATE_LIMIT = 5  # requests per second
RATE_LIMIT_WINDOW = 1  # seconds

# --- Database Initialization ---
def init_db():
    conn = sqlite3.connect('agentforge.db')
    c = conn.cursor()
    
    # Create tables if they don't exist
    c.execute('''CREATE TABLE IF NOT EXISTS agents (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        model TEXT NOT NULL,
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        performance_score REAL DEFAULT 0.0,
        total_conversations INTEGER DEFAULT 0,
        embed_enabled BOOLEAN DEFAULT 0
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS knowledge_sources (
        id TEXT PRIMARY KEY,
        agent_id TEXT,
        source_type TEXT NOT NULL,
        content TEXT,
        filename TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(agent_id) REFERENCES agents(id)
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        agent_id TEXT,
        session_id TEXT,
        user_message TEXT,
        ai_response TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        rating INTEGER,
        FOREIGN KEY(agent_id) REFERENCES agents(id)
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS embeddings (
        agent_id TEXT PRIMARY KEY,
        faiss_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(agent_id) REFERENCES agents(id)
    )''')
    
    # Check for missing columns and add them
    try:
        c.execute("PRAGMA table_info(agents)")
        columns = [col[1] for col in c.fetchall()]
        
        # Add missing columns
        if 'performance_score' not in columns:
            c.execute("ALTER TABLE agents ADD COLUMN performance_score REAL DEFAULT 0.0")
        if 'total_conversations' not in columns:
            c.execute("ALTER TABLE agents ADD COLUMN total_conversations INTEGER DEFAULT 0")
        if 'embed_enabled' not in columns:
            c.execute("ALTER TABLE agents ADD COLUMN embed_enabled BOOLEAN DEFAULT 0")
            
        # Check knowledge_sources table
        c.execute("PRAGMA table_info(knowledge_sources)")
        columns = [col[1] for col in c.fetchall()]
        if 'filename' not in columns:
            c.execute("ALTER TABLE knowledge_sources ADD COLUMN filename TEXT")
            
    except sqlite3.OperationalError as e:
        print(f"Database migration error: {str(e)}")
    
    conn.commit()
    conn.close()

init_db()

# --- Initialize Embeddings ---
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# --- Helper Functions ---
def get_db():
    conn = sqlite3.connect('agentforge.db')
    conn.row_factory = sqlite3.Row
    return conn

def sanitize_input(input_str: str) -> str:
    """Sanitize user input to prevent XSS and SQL injection"""
    if not input_str:
        return ""
    # Remove potentially harmful characters
    sanitized = re.sub(r"[;\\\"\']", "", input_str)
    return html.escape(sanitized)

def rate_limit(request: Request):
    """Simple rate limiting implementation"""
    client_ip = request.client.host
    now = time.time()
    
    if client_ip not in rate_limits:
        rate_limits[client_ip] = []
    
    # Remove old requests
    rate_limits[client_ip] = [t for t in rate_limits[client_ip] if t > now - RATE_LIMIT_WINDOW]
    
    if len(rate_limits[client_ip]) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests")
    
    rate_limits[client_ip].append(now)

def process_text(text: str) -> List[Document]:
    """Improved text processing with better chunking strategy"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
    )
    return splitter.create_documents([text])

def process_pdf(file: UploadFile) -> List[Document]:
    """Improved PDF processing using PyPDF2"""
    try:
        content = file.file.read()
        if not content:
            raise ValueError("Empty PDF file")
            
        # Create a BytesIO object
        pdf_file = io.BytesIO(content)
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        return process_text(text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF processing error: {str(e)}")

def process_url(url: str) -> List[Document]:
    """Enhanced URL processing with content validation"""
    try:
        # Validate URL format
        if not re.match(r"^https?://", url):
            raise ValueError("Invalid URL format")
            
        response = requests.get(url, timeout=15, headers={"User-Agent": "AgentForge/1.0"})
        response.raise_for_status()
        
        # Check if content is HTML
        if "text/html" in response.headers.get("Content-Type", ""):
            # Extract main content
            text = re.sub("<[^<]+?>", " ", response.text)
            text = re.sub(r"\s+", " ", text).strip()
        else:
            text = response.text
            
        return process_text(text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching URL: {str(e)}")

def create_vectorstore(agent_id: str, docs: List[Document]):
    """Create and cache vector store"""
    if not docs:
        raise ValueError("No documents to create vector store")
        
    vectorstore = FAISS.from_documents(docs, embeddings)
    path = f"vectorstores/{agent_id}"
    vectorstore.save_local(path)
    
    with get_db() as conn:
        conn.execute("INSERT OR REPLACE INTO embeddings (agent_id, faiss_path) VALUES (?, ?)", 
                    (agent_id, path))
        conn.commit()
    
    vectorstore_cache[agent_id] = vectorstore
    return vectorstore

def get_vectorstore(agent_id: str):
    """Get vector store with caching"""
    if agent_id in vectorstore_cache:
        return vectorstore_cache[agent_id]
    
    with get_db() as conn:
        row = conn.execute("SELECT faiss_path FROM embeddings WHERE agent_id = ?", 
                          (agent_id,)).fetchone()
    
    if not row:
        return None
    
    try:
        vectorstore = FAISS.load_local(row["faiss_path"], embeddings, 
                                      allow_dangerous_deserialization=True)
        vectorstore_cache[agent_id] = vectorstore
        return vectorstore
    except Exception as e:
        print(f"Error loading vectorstore: {str(e)}")
        return None

def create_llm(model: str):
    """Create LLM with API key validation"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model,
        temperature=0.3,  # More deterministic responses
        max_tokens=2048,
        max_retries=3,
        request_timeout=30
    )

def create_qa_chain(agent_id: str, model: str, agent_name: str):
    """Create QA chain with caching and improved prompts"""
    cache_key = f"{agent_id}-{model}"
    if cache_key in chain_cache:
        return chain_cache[cache_key]
    
    vectorstore = get_vectorstore(agent_id)
    
    # If no vectorstore exists, create a conversation chain
    if not vectorstore:
        memory = ConversationBufferWindowMemory(
            k=7, memory_key="chat_history", return_messages=True
        )
        
        template = f"""You are {agent_name}, an AI assistant designed to help users. 
Be knowledgeable, helpful, and concise. 
If you don't know the answer to a question, honestly say you don't know. Do not make up information.

Current conversation:
{{chat_history}}
Human: {{input}}
Assistant:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["chat_history", "input"]
        )
        
        chain = ConversationChain(
            llm=create_llm(model),
            memory=memory,
            prompt=prompt
        )
    else:
        # Create RAG chain with explicit output key
        memory = ConversationBufferWindowMemory(
            k=7,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"  # Explicitly specify which key to store
        )
        
        template = f"""You are {agent_name}, an AI assistant. 
Use the provided context and chat history to answer the user's question.
If the context does not provide enough information, use your own knowledge to answer but clearly state that the information is not from the provided context.

Context:
{{context}}

Chat History:
{{chat_history}}

Question: {{question}}

Answer in a clear, concise, and accurate manner:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "chat_history", "question"]
        )
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=create_llm(model),
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
            output_key="answer"  # Explicit output key
        )
    
    chain_cache[cache_key] = chain
    return chain

def update_agent_performance(agent_id: str, rating: int = None):
    with get_db() as conn:
        conn.execute("UPDATE agents SET total_conversations = total_conversations + 1 WHERE id = ?", 
                    (agent_id,))
        
        if rating:
            avg_rating = conn.execute(
                "SELECT AVG(rating) FROM conversations WHERE agent_id = ? AND rating IS NOT NULL", 
                (agent_id,)
            ).fetchone()[0]
            
            if avg_rating:
                conn.execute("UPDATE agents SET performance_score = ? WHERE id = ?", 
                           (round(avg_rating, 2), agent_id))
        
        conn.commit()

# --- Google Drive Functions ---
def get_google_credentials(token_path: str):
    creds = None
    if os.path.exists(token_path):
        try:
            creds = Credentials.from_authorized_user_file(token_path, GOOGLE_SCOPES)
        except Exception as e:
            print(f"Error loading credentials: {str(e)}")
    return creds

def save_google_credentials(creds: Credentials, token_path: str):
    with open(token_path, 'w') as token:
        token.write(creds.to_json())

def get_google_drive_service(token: str = None):
    token_path = f"google_tokens/{token}.json" if token else "google_tokens/default.json"
    creds = get_google_credentials(token_path)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(google.auth.transport.requests.Request())
            except Exception as e:
                print(f"Error refreshing credentials: {str(e)}")
        else:
            if not os.path.exists("credentials.json"):
                raise HTTPException(
                    status_code=400, 
                    detail="Google credentials file not found. Place credentials.json in the root directory."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", GOOGLE_SCOPES
            )
            creds = flow.run_local_server(port=0)
        save_google_credentials(creds, token_path)
    
    return build('drive', 'v3', credentials=creds)

# --- API Routes ---
@app.get("/")
async def root():
    return {"message": "AgentForge API", "docs": "/docs"}

@app.get("/ui", response_class=HTMLResponse)
async def get_ui():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>UI not found. Please create static/index.html</h1>", status_code=404)

@app.post("/agents", response_model=Agent)
async def create_agent(
    request: Request,
    name: str = Form(...),
    model: str = Form(...),
    description: Optional[str] = Form(None),
    embed_enabled: bool = Form(False)
):
    rate_limit(request)
    
    # Sanitize inputs
    name = sanitize_input(name)
    model = sanitize_input(model)
    description = sanitize_input(description) if description else None
    
    if model not in GROQ_MODELS:
        raise HTTPException(status_code=400, detail="Invalid model")
    
    agent_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    
    with get_db() as conn:
        try:
            conn.execute(
                "INSERT INTO agents (id, name, model, description, created_at, embed_enabled) VALUES (?, ?, ?, ?, ?, ?)",
                (agent_id, name, model, description, created_at, int(embed_enabled)))
            conn.commit()
            
            row = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
            return Agent(**dict(row))
        except sqlite3.OperationalError as e:
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/agents", response_model=List[Agent])
async def list_agents(request: Request):
    rate_limit(request)
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM agents ORDER BY created_at DESC").fetchall()
    return [Agent(**dict(row)) for row in rows]

@app.get("/agents/{agent_id}", response_model=Agent)
async def get_agent(request: Request, agent_id: str):
    rate_limit(request)
    agent_id = sanitize_input(agent_id)
    
    with get_db() as conn:
        row = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
    
    if not row:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return Agent(**dict(row))

@app.patch("/agents/{agent_id}", response_model=Agent)
async def update_agent(request: Request, agent_id: str, update_data: AgentUpdate):
    rate_limit(request)
    agent_id = sanitize_input(agent_id)
    
    # Sanitize update data
    if update_data.name:
        update_data.name = sanitize_input(update_data.name)
    if update_data.model:
        update_data.model = sanitize_input(update_data.model)
    if update_data.description:
        update_data.description = sanitize_input(update_data.description)
    
    with get_db() as conn:
        agent = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        if update_data.model and update_data.model not in GROQ_MODELS:
            raise HTTPException(status_code=400, detail="Invalid model")
        
        # Build update query
        updates = []
        params = []
        if update_data.name is not None:
            updates.append("name = ?")
            params.append(update_data.name)
        if update_data.model is not None:
            updates.append("model = ?")
            params.append(update_data.model)
        if update_data.description is not None:
            updates.append("description = ?")
            params.append(update_data.description)
        if update_data.embed_enabled is not None:
            updates.append("embed_enabled = ?")
            params.append(int(update_data.embed_enabled))
        
        if not updates:
            return Agent(**dict(agent))
        
        query = f"UPDATE agents SET {', '.join(updates)} WHERE id = ?"
        params.append(agent_id)
        conn.execute(query, params)
        conn.commit()
        
        # Clear caches
        cache_keys = [key for key in chain_cache.keys() if key.startswith(agent_id)]
        for key in cache_keys:
            chain_cache.pop(key, None)
        
        if agent_id in vectorstore_cache:
            vectorstore_cache.pop(agent_id)
        
        # Fetch updated agent
        updated_agent = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
        return Agent(**dict(updated_agent))

@app.delete("/agents/{agent_id}")
async def delete_agent(request: Request, agent_id: str):
    rate_limit(request)
    agent_id = sanitize_input(agent_id)
    
    with get_db() as conn:
        result = conn.execute("DELETE FROM agents WHERE id = ?", (agent_id,)).rowcount
        if result == 0:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        conn.execute("DELETE FROM knowledge_sources WHERE agent_id = ?", (agent_id,))
        conn.execute("DELETE FROM conversations WHERE agent_id = ?", (agent_id,))
        conn.execute("DELETE FROM embeddings WHERE agent_id = ?", (agent_id,))
        conn.commit()
    
    # Clear caches
    cache_keys = [key for key in chain_cache.keys() if key.startswith(agent_id)]
    for key in cache_keys:
        chain_cache.pop(key, None)
    
    if agent_id in vectorstore_cache:
        vectorstore_cache.pop(agent_id)
    
    # Delete vectorstore files
    try:
        os.remove(f"vectorstores/{agent_id}/index.faiss")
        os.remove(f"vectorstores/{agent_id}/index.pkl")
    except FileNotFoundError:
        pass
    
    return {"message": "Agent deleted successfully"}

@app.post("/agents/{agent_id}/knowledge")
async def upload_knowledge(
    request: Request,
    agent_id: str,
    source_type: str = Form(...),
    file: UploadFile = File(None),
    url: str = Form(None),
    text: str = Form(None)
):
    rate_limit(request)
    agent_id = sanitize_input(agent_id)
    source_type = sanitize_input(source_type)
    url = sanitize_input(url) if url else None
    text = sanitize_input(text) if text else None
    
    with get_db() as conn:
        agent = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
    
    docs = []
    content_desc = ""
    filename = None
    
    try:
        if source_type == "pdf" and file:
            if file.filename and not file.filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are accepted")
            docs = process_pdf(file)
            content_desc = f"PDF: {file.filename}"
            filename = file.filename
        elif source_type == "url" and url:
            docs = process_url(url)
            content_desc = f"URL: {url}"
        elif source_type == "text" and text:
            docs = process_text(text)
            content_desc = f"Text: {text[:200]}..." if len(text) > 200 else f"Text: {text}"
        else:
            raise HTTPException(status_code=400, detail="Invalid source type or missing content")
        
        if not docs:
            raise HTTPException(status_code=400, detail="No content could be extracted")
        
        # Update or create vector store
        existing_vectorstore = get_vectorstore(agent_id)
        if existing_vectorstore:
            existing_vectorstore.add_documents(docs)
            path = f"vectorstores/{agent_id}"
            existing_vectorstore.save_local(path)
            vectorstore_cache[agent_id] = existing_vectorstore
        else:
            create_vectorstore(agent_id, docs)
        
        # Save knowledge source
        source_id = str(uuid.uuid4())
        with get_db() as conn:
            conn.execute(
                "INSERT INTO knowledge_sources (id, agent_id, source_type, content, filename) VALUES (?, ?, ?, ?, ?)",
                (source_id, agent_id, source_type, content_desc, filename)
            )
            conn.commit()
        
        # Clear chain cache for this agent
        cache_keys = [key for key in chain_cache.keys() if key.startswith(agent_id)]
        for key in cache_keys:
            chain_cache.pop(key, None)
        
        return {"message": "Knowledge source added successfully", "source_id": source_id}
    
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/agents/{agent_id}/knowledge")
async def get_knowledge_sources(request: Request, agent_id: str):
    rate_limit(request)
    agent_id = sanitize_input(agent_id)
    
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM knowledge_sources WHERE agent_id = ? ORDER BY created_at DESC", 
            (agent_id,)
        ).fetchall()
    return [dict(row) for row in rows]

@app.delete("/agents/{agent_id}/knowledge/{source_id}")
async def delete_knowledge_source(request: Request, agent_id: str, source_id: str):
    rate_limit(request)
    agent_id = sanitize_input(agent_id)
    source_id = sanitize_input(source_id)
    
    with get_db() as conn:
        source = conn.execute(
            "SELECT * FROM knowledge_sources WHERE id = ? AND agent_id = ?",
            (source_id, agent_id)
        ).fetchone()
        
        if not source:
            raise HTTPException(status_code=404, detail="Knowledge source not found")
        
        conn.execute(
            "DELETE FROM knowledge_sources WHERE id = ? AND agent_id = ?",
            (source_id, agent_id)
        )
        
        # Rebuild vectorstore without this source
        sources = conn.execute(
            "SELECT * FROM knowledge_sources WHERE agent_id = ?",
            (agent_id,)
        ).fetchall()
        
        if sources:
            docs = []
            for src in sources:
                if src["source_type"] == "text":
                    docs.extend(process_text(src["content"]))
            create_vectorstore(agent_id, docs)
        else:
            conn.execute("DELETE FROM embeddings WHERE agent_id = ?", (agent_id,))
            try:
                os.remove(f"vectorstores/{agent_id}/index.faiss")
                os.remove(f"vectorstores/{agent_id}/index.pkl")
            except FileNotFoundError:
                pass
        
        conn.commit()
    
    # Clear caches
    cache_keys = [key for key in chain_cache.keys() if key.startswith(agent_id)]
    for key in cache_keys:
        chain_cache.pop(key, None)
    
    if agent_id in vectorstore_cache:
        vectorstore_cache.pop(agent_id)
    
    return {"message": "Knowledge source deleted successfully"}

@app.post("/agents/{agent_id}/chat", response_model=ConversationResponse)
async def chat_with_agent(request: Request, agent_id: str, conversation: ConversationCreate):
    rate_limit(request)
    agent_id = sanitize_input(agent_id)
    conversation.message = sanitize_input(conversation.message)
    conversation.session_id = sanitize_input(conversation.session_id)
    
    with get_db() as conn:
        agent = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
    
    try:
        qa_chain = create_qa_chain(agent_id, agent["model"], agent["name"])
        
        # Check chain type
        if hasattr(qa_chain, 'retriever'):
            result = qa_chain.invoke({
                "question": conversation.message
            })
            response = result["answer"]
        else:
            result = qa_chain.invoke({
                "input": conversation.message
            })
            response = result["response"]
        
        # Save conversation
        with get_db() as conn:
            conn.execute(
                "INSERT INTO conversations (agent_id, session_id, user_message, ai_response) VALUES (?, ?, ?, ?)",
                (agent_id, conversation.session_id, conversation.message, response)
            )
            conn.commit()
        
        # Update performance
        update_agent_performance(agent_id)
        
        return ConversationResponse(
            response=response,
            agent_id=agent_id,
            session_id=conversation.session_id
        )
    
    except Exception as e:
        print(f"Chat error: {str(e)}")
        return ConversationResponse(
            response="I'm having trouble processing your request right now. Please try again later.",
            agent_id=agent_id,
            session_id=conversation.session_id
        )

@app.post("/conversations/{conversation_id}/rate")
async def rate_conversation(request: Request, conversation_id: int, rating: int = Form(..., ge=1, le=5)):
    rate_limit(request)
    with get_db() as conn:
        result = conn.execute(
            "UPDATE conversations SET rating = ? WHERE id = ?",
            (rating, conversation_id)
        ).rowcount
        
        if result == 0:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get agent ID for performance update
        row = conn.execute(
            "SELECT agent_id FROM conversations WHERE id = ?",
            (conversation_id,)
        ).fetchone()
        
        if row:
            update_agent_performance(row["agent_id"], rating)
        
        conn.commit()
    
    return {"message": "Rating saved successfully"}

@app.get("/agents/{agent_id}/conversations")
async def get_conversations(request: Request, agent_id: str, session_id: Optional[str] = None):
    rate_limit(request)
    agent_id = sanitize_input(agent_id)
    
    with get_db() as conn:
        query = "SELECT * FROM conversations WHERE agent_id = ?"
        params = [agent_id]
        
        if session_id:
            query += " AND session_id = ?"
            params.append(sanitize_input(session_id))
            
        query += " ORDER BY timestamp DESC"
        
        rows = conn.execute(query, params).fetchall()
    
    return [dict(row) for row in rows]

@app.get("/gdrive/status")
async def google_drive_status(request: Request):
    rate_limit(request)
    try:
        service = get_google_drive_service()
        return {"connected": True}
    except:
        return {"connected": False}

@app.get("/gdrive/auth")
async def google_drive_auth(request: Request):
    rate_limit(request)
    try:
        service = get_google_drive_service()
        return RedirectResponse(url="/gdrive/files")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Google Drive connection failed: {str(e)}")

@app.get("/gdrive/files")
async def get_google_drive_files(request: Request, query: str = None):
    rate_limit(request)
    try:
        service = get_google_drive_service()
        results = service.files().list(
            q=query if query else "",
            pageSize=20,
            fields="files(id, name, mimeType, modifiedTime, size)"
        ).execute()
        return {"files": results.get("files", [])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching files: {str(e)}")

@app.post("/gdrive/import")
async def import_google_drive_file(
    request: Request,
    file_id: str = Form(...),
    agent_id: str = Form(...)
):
    rate_limit(request)
    agent_id = sanitize_input(agent_id)
    file_id = sanitize_input(file_id)
    
    try:
        service = get_google_drive_service()
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while not done:
            _, done = downloader.next_chunk()
        
        fh.seek(0)
        file_content = fh.read()
        
        # Create a temporary file-like object
        file_like = io.BytesIO(file_content)
        file_like.name = "google_drive_file.pdf"
        
        # Create an UploadFile instance
        file = UploadFile(file=file_like, filename="google_drive_file.pdf")
        
        # Process as PDF
        docs = process_pdf(file)
        
        # Update vector store
        existing_vectorstore = get_vectorstore(agent_id)
        if existing_vectorstore:
            existing_vectorstore.add_documents(docs)
            path = f"vectorstores/{agent_id}"
            existing_vectorstore.save_local(path)
            vectorstore_cache[agent_id] = existing_vectorstore
        else:
            create_vectorstore(agent_id, docs)
        
        # Save knowledge source
        source_id = str(uuid.uuid4())
        with get_db() as conn:
            conn.execute(
                "INSERT INTO knowledge_sources (id, agent_id, source_type, content, filename) VALUES (?, ?, ?, ?, ?)",
                (source_id, agent_id, "pdf", f"Google Drive File: {file_id}", "google_drive_file.pdf")
            )
            conn.commit()
        
        # Clear chain cache
        cache_keys = [key for key in chain_cache.keys() if key.startswith(agent_id)]
        for key in cache_keys:
            chain_cache.pop(key, None)
        
        return {"status": "success", "message": "Google Drive file added to knowledge base"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing Google Drive file: {str(e)}")

@app.get("/models")
async def get_models(request: Request):
    rate_limit(request)
    return {"models": GROQ_MODELS}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/agents/{agent_id}/embed")
async def get_embed_code(request: Request, agent_id: str):
    rate_limit(request)
    agent_id = sanitize_input(agent_id)
    
    with get_db() as conn:
        agent = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
        if not agent or not agent["embed_enabled"]:
            raise HTTPException(status_code=404, detail="Embedding not enabled for this agent")
    
    base_url = str(request.base_url).rstrip('/')
    embed_url = f"{base_url}/agents/{agent_id}/embed/chat"
    
    embed_code = f"""
    <iframe 
        src="{embed_url}" 
        width="100%" 
        height="500px" 
        style="border:1px solid #ddd; border-radius:5px;"
        allow="clipboard-write"
    >
    </iframe>
    """
    
    return {"iframe_code": embed_code}

@app.get("/agents/{agent_id}/embed/chat", response_class=HTMLResponse)
async def embedded_chat_ui(request: Request, agent_id: str):
    try:
        with open("static/embed.html", "r", encoding="utf-8") as f:
            content = f.read().replace("{{AGENT_ID}}", agent_id)
            return HTMLResponse(content)
    except FileNotFoundError:
        return HTMLResponse("<h1>Embedded chat UI not found</h1>", status_code=404)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8230, timeout_keep_alive=30)