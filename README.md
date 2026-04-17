# RAG Document QA API

A powerful **Conversational RAG (Retrieval-Augmented Generation)** API that allows users to upload a PDF document and ask intelligent questions about it with full conversation memory.

Built with **FastAPI**, **LangChain**, **Groq (Llama-3.1)**, and **ChromaDB**.

---

## ✨ Features

- Upload PDF documents and process them instantly
- Conversational chat with **memory** (follow-up questions work naturally)
- Returns relevant sources and context used for each answer
- Fast inference using **Groq** (Llama-3.1)
- Clean REST API with interactive Swagger UI
- Session-based memory management

## 🛠️ Tech Stack

- **Framework**: FastAPI
- **LLM**: Groq (Llama-3.1)
- **Orchestration**: LangChain
- **Vector Database**: Chroma
- **Embeddings**: Hugging Face / Groq compatible
- **Document Loader**: PyPDF

## 📡 API Endpoints

| Method | Endpoint           | Description                              |
|--------|--------------------|------------------------------------------|
| GET    | `/`                | Health check & welcome message           |
| POST   | `/process-pdf`     | Upload PDF and create a new chat session |
| POST   | `/chat`            | Send question and get answer (with memory) |

---

## 🚀 Local Setup & Run

### 1. Clone the repository

```bash
git clone https://github.com/akshaysrsh3302-tech/rag-ai-agent.git
cd rag-ai-agent
