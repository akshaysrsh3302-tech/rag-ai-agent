# RAG Document QA API - FastAPI + LangChain + Groq

A conversational RAG (Retrieval-Augmented Generation) API that allows users to upload a PDF and ask questions with conversation memory.

## Live Deployment
**URL:** https://your-app-name.onrender.com   ← Update this after deployment

## Features
- Upload PDF documents
- Conversational chat with memory (follow-up questions work)
- Returns relevant sources/context
- Built with FastAPI, LangChain, Groq (Llama-3.1), and Chroma

## Endpoints
- `GET /` → Health check
- `POST /process-pdf` → Upload PDF and create session
- `POST /chat` → Ask questions (requires session_id)

## Local Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR-USERNAME/rag-ai-agent.git
cd rag-ai-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Groq API key
export GROQ_API_KEY="gsk_YourKeyHere"

# 4. Run the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
