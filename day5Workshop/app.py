import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()

# Initialize FastAPI application
api = FastAPI()

# -----------------------------
# Embedding Model Setup
# -----------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Vector Store (Chroma DB)
# -----------------------------
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

# -----------------------------
# Language Model (Groq)
# -----------------------------
chat_model = ChatGroq(
    model_name="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# -----------------------------
# Request Schema
# -----------------------------
class QueryInput(BaseModel):
    question: str


# -----------------------------
# Upload PDF Endpoint
# -----------------------------
@api.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    temp_path = "uploaded_doc.pdf"

    # Save uploaded file
    with open(temp_path, "wb") as pdf_file:
        pdf_file.write(await file.read())

    # Load PDF content
    pdf_loader = PyPDFLoader(temp_path)
    documents = pdf_loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )

    document_chunks = text_splitter.split_documents(documents)

    # Store chunks in vector database
    vector_store.add_documents(document_chunks)

    return {"status": "success", "message": "PDF processed and stored."}


# -----------------------------
# Question Answering Endpoint
# -----------------------------
@api.post("/ask")
def ask(request: QueryInput):

    # Retrieve relevant chunks
    results = vector_store.similarity_search(request.question, k=3)

    if len(results) == 0:
        return {
            "question": request.question,
            "answer": "No matching information found in the documents."
        }

    # Prepare context from retrieved documents
    context = "\n\n---\n\n".join(
        [f"Source {index + 1}:\n{doc.page_content}" for index, doc in enumerate(results)]
    )

    # System instruction
    instruction = (
        "You are an AI assistant that answers questions using ONLY the given document sources. "
        "If the answer cannot be found in the sources, respond with 'Not in document'. "
        "Include the source numbers in your answer."
    )

    # User message
    prompt = f"Context:\n{context}\n\nQuestion: {request.question}"

    messages = [
        SystemMessage(content=instruction),
        HumanMessage(content=prompt)
    ]

    # Generate response
    reply = chat_model.invoke(messages)

    return {
        "question": request.question,
        "answer": reply.content
    }