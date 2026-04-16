import os
import uuid
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory

app = FastAPI(title="RAG Document QA API")

# Get Groq API Key from Render Environment Variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set!")

llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant", temperature=0.3)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

sessions = {}

class ChatRequest(BaseModel):
    session_id: str
    question: str

@app.post("/process-pdf")
async def process_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        file_path = tmp.name

    try:
        loader = PyPDFLoader(file_path)
        raw_documents = loader.load()

        # Add metadata summary
        metadata_summary = "Document Metadata:\n"
        if raw_documents:
            meta = raw_documents[0].metadata
            title = meta.get('title', 'Unknown Title')
            author = meta.get('author', 'Unknown Author')
            metadata_summary += f"Title: {title}\nAuthors: {author}\n"
            first_page = raw_documents[0].page_content.strip()
            raw_documents[0].page_content = metadata_summary + "\n\n" + first_page

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
        splits = text_splitter.split_documents(raw_documents)

        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

        system_prompt = (
            "You are a precise document QA assistant. "
            "Use the Document Metadata at the beginning when asked about title or authors. "
            "Answer ONLY based on the retrieved context. "
            "If you cannot find the information, say so honestly.\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given the chat history and the latest user question, formulate a standalone question."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        session_history = ChatMessageHistory()
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

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
        if os.path.exists(file_path):
            os.unlink(file_path)

@app.post("/chat")
def chat(request: ChatRequest):
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Upload a PDF first.")

    chain = sessions[request.session_id]["chain"]

    response = chain.invoke(
        {"input": request.question},
        config={"configurable": {"session_id": request.session_id}}
    )

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

@app.get("/")
def home():
    return {"message": "RAG AI Agent is running! Visit /docs for Swagger UI"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
