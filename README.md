# 🚀 aGi Claims AI: Local RAG Pipeline

**An end-to-end applied AI solution for transforming unstructured insurance data into structured technical reports.**

## Live Demos

- [Flowchat UI](https://abish-gupta.github.io/aGi-claims-rag-pipeline/FlowChart/)
- [Frontend UI](https://abish-gupta.github.io/aGi-claims-rag-pipeline/Frontend/)

## 🧠 The Philosophy: Build vs. Buy
In modern AI architecture, writing custom ML models for basic document parsing is a waste of engineering hours. This pipeline operates on a strict Build vs. Buy protocol:
1. **Buy/Integrate:** Use managed OCR (like Azure Document Intelligence) to handle messy, unstructured PDFs.
2. **Build:** Construct a custom, locally-hosted Retrieval-Augmented Generation (RAG) pipeline to control business logic, guarantee data privacy, and prevent LLM hallucinations.

## 🏗️ The Architecture Story (How it Works)

### 🌊 Phase 1: The Messy Reality (Ingestion)
Insurance claims rely on 10-year-old PDFs and messy site reports. We don't feed this blindly to an LLM. We extract the text and slice it into **overlapping chunks**. 
* **Why?** LLMs have strict context windows. Slicing prevents the AI from forgetting crucial damage reports or cutting a sentence in half.

### 🛡️ Phase 2: The Vault (Vectorization & Privacy)
We convert those text chunks into mathematical vectors using **open-source embedding models** (HuggingFace) and store them in a local Vector Database (FAISS). 
* **Why?** Absolute privacy. Sensitive insurance data never leaves the company's VPC. Zero exposure to public APIs like OpenAI.

### ⚡ Phase 3: The Brain (Retrieval & Generation)
When a C# backend queries the system via our FastAPI endpoint, we don't ask the LLM to guess. We run a semantic search, grab the top 3 most relevant factual chunks, and feed *only* those to a **locally-hosted LLM** (Llama 3/Mistral).
* **Why?** It forces the model to act as a strict extraction engine based entirely on facts, dropping hallucination rates to near zero.

## 💻 Tech Stack
* **Backend Integration:** Python, FastAPI (Decoupled REST API ready for C# consumption)
* **AI Engine:** LangChain, FAISS (Vector DB), Ollama (Local LLM), HuggingFace (Embeddings)
* **Frontend Demo:** HTML5, TailwindCSS, Vanilla JS

## 🚀 Run the Project
1. Run backend: `cd backend && uvicorn main:app --reload`
2. Open `frontend/index.html` in your browser.
