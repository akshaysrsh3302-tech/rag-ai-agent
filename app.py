import nest_asyncio
from pyngrok import ngrok
import uvicorn
import os
import uuid
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain imports
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="RAG Document QA API")

# ====================== CONFIG FROM .env ======================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NGROK_AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY is not set in .env file")

if NGROK_AUTHTOKEN:
    ngrok.set_auth_token(NGROK_AUTHTOKEN)
    print("✅ ngrok auth token loaded successfully")
else:
    print("⚠️  NGROK_AUTHTOKEN not found in .env. ngrok may not work.")

# Initialize LLM and Embeddings
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY, 
    model_name="llama-3.1-8b-instant", 
    temperature=0.3
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Store active sessions: session_id → {chain, vectorstore, history}
sessions = {}

class ChatRequest(BaseModel):
    session_id: str
    question: str


@app.get("/")
async def home():
    return {
        "message": "RAG Document QA API is running successfully!",
        "docs_url": "/docs",
        "instruction": "First upload PDF using /process-pdf, then use /chat"
    }


@app.post("/process-pdf")
async def process_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        file_path = tmp.name

    try:
        # Load and split PDF
        loader = PyPDFLoader(file_path)
        raw_documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
        splits = text_splitter.split_documents(raw_documents)

        # Create vector store
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

        # Prompts
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given the chat history and the latest user question, formulate a standalone question."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer only based on the provided context.\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # Create chains
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Add memory
        session_history = ChatMessageHistory()

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        # Save session
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "chain": conversational_rag_chain,
            "vectorstore": vectorstore,
            "history": session_history
        }

        return {
            "session_id": session_id,
            "message": "PDF processed successfully",
            "num_chunks": len(splits),
            "filename": file.filename
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(file_path):
            os.unlink(file_path)


@app.post("/chat")
async def chat(request: ChatRequest):
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a PDF first.")

    chain = sessions[request.session_id]["chain"]

    response = chain.invoke(
        {"input": request.question},
        config={"configurable": {"session_id": request.session_id}}
    )

    # Extract sources
    sources = []
    for i, doc in enumerate(response.get("context", []), 1):
        content = doc.page_content.strip()
        if len(content) > 600:
            content = content[:600] + "..."
        sources.append({"source_id": i, "content": content})

    return {
        "answer": response.get("answer", "Sorry, I could not generate an answer."),
        "sources": sources,
        "num_sources": len(sources)
    }


# ====================== Run the Server ======================
if __name__ == "__main__":
    PORT = 8000

    # Start ngrok tunnel
    try:
        ngrok_tunnel = ngrok.connect(PORT)
        print("\n🚀 **Your Public API URL:**", ngrok_tunnel.public_url)
        print("📋 **Swagger UI (Recommended):**", ngrok_tunnel.public_url + "/docs")
        print("🔗 **Chat Endpoint:**", ngrok_tunnel.public_url + "/chat\n")
    except Exception as e:
        print(f"⚠️ Could not start ngrok: {e}")
        print("Make sure NGROK_AUTHTOKEN is set in .env file\n")

    print("✅ Server is starting...\n")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
