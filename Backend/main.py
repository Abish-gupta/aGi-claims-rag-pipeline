from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

# --- LangChain & AI Imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- API Setup ---
app = FastAPI(title="aGi Claims AI Engine - Production Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Configurations ---
DB_PATH = "local_faiss_index"
EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
VECTOR_DB = None

# --- Payloads ---
class QueryRequest(BaseModel):
    query: str
    filename: str

class IngestRequest(BaseModel):
    filename: str

# --- Core Logic ---
def load_or_initialize_db(file_path: str = None):
    global VECTOR_DB
    
    # Check if we already have a saved persistent database
    if os.path.exists(DB_PATH):
        print("🗄️ Loading existing persistent Vector DB from disk...")
        # allow_dangerous_deserialization is required in newer Langchain versions for local FAISS loads
        VECTOR_DB = FAISS.load_local(DB_PATH, EMBEDDINGS, allow_dangerous_deserialization=True)
        return VECTOR_DB
    
    # If no DB exists, we must ingest a file
    if not file_path or not os.path.exists(file_path):
        raise ValueError("No database exists and no valid file provided to build one.")

    print(f"📄 No existing DB found. Extracting messy data from: {file_path}")
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # The Build vs. Buy chunking strategy
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    print("🧠 Building and saving new local Vector DB...")
    VECTOR_DB = FAISS.from_documents(splits, EMBEDDINGS)
    VECTOR_DB.save_local(DB_PATH) # The flex: Saving to disk for persistence
    return VECTOR_DB

# Initialize the DB on startup if it already exists
if os.path.exists(DB_PATH):
    load_or_initialize_db()

# --- Endpoints ---

@app.post("/api/ingest")
async def ingest_document(request: IngestRequest):
    """Endpoint for the C# backend to upload new claim documents into the Vector DB."""
    try:
        load_or_initialize_db(request.filename)
        return {"status": "success", "message": f"Document {request.filename} successfully vectorized and stored locally."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate")
async def generate_report(request: QueryRequest):
    """Endpoint to query the Vector DB and generate the technical report."""
    global VECTOR_DB
    
    if VECTOR_DB is None:
        try:
            load_or_initialize_db(request.filename)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    print("⚙️ Generating report using local Llama 3...")
    llm = Ollama(model="llama3") 
    
    system_prompt = (
        "You are an expert AI extraction agent for insurance claims. "
        "Use the provided context to answer the query. "
        "Format your output as a structured technical report in Markdown. "
        "If you don't know the answer, strictly state that the data is missing to prevent hallucinations.\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(VECTOR_DB.as_retriever(search_kwargs={"k": 3}), question_answer_chain)
    
    response = rag_chain.invoke({"input": request.query})
    
    return {"report": response["answer"]}
