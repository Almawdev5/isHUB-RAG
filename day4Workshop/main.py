import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_groq import ChatGroq


# ============================================
# 1️⃣ Load Environment Variables
# ============================================
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")


# ============================================
# 2️⃣ Load CV PDF
# ============================================
print("📄 Loading CV...")

loader = PyPDFLoader("Image.data/CV.pdf")
documents = loader.load()


# ============================================
# 3️⃣ Split Document into Chunks
# ============================================
print("✂ Splitting document into semantic chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)


# ============================================
# 4️⃣ Create Embeddings
# ============================================
print("🧠 Generating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================
# 5️⃣ Store in Chroma Vector Database
# ============================================
print("📦 Creating vector database...")

db = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./chroma_db"
)

retriever = db.as_retriever(search_kwargs={"k": 3})


# ============================================
# 6️⃣ Initialize LLM (Groq - LLaMA 3.1)
# ============================================
print("⚡ Connecting to LLM...")

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    groq_api_key=groq_api_key
)


# ============================================
# 7️⃣ Professional System Prompt
# ============================================

SYSTEM_PROMPT = """
You are a Professional AI CV Assistant.

Your task is to answer questions strictly based on the provided CV content.

=============================
STRICT RULES
=============================
1. Use ONLY the provided CV context.
2. Do NOT use prior knowledge.
3. Do NOT assume or infer missing details.
4. Do NOT generate external information.
5. If the answer is not explicitly written in the context, respond exactly with:

"The requested information is not available in the provided CV content."

=============================
RESPONSE FORMAT RULES
=============================
• Answer using bullet points only.
• Keep responses concise and professional.
• Highlight important elements using symbols:

  📌  Key Information
  🎓  Education
  💼  Experience
  🛠  Skills
  🚀  Achievements

• Do NOT add explanations beyond what is written.
• Maintain a confident and professional tone.
"""


print("\n🚀 CV RAG Assistant Ready!\n")


# ============================================
# 8️⃣ Interactive Question Loop
# ============================================
while True:
    query = input("Ask about the CV (or type 'exit'): ")

    if query.lower() == "exit":
        print("\n👋 Exiting CV Assistant.")
        break

    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(query)

    # Combine retrieved content into context
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Build final prompt
    prompt = f"""
{SYSTEM_PROMPT}

-----------------------------
CV CONTEXT:
{context}
-----------------------------

USER QUESTION:
{query}

FINAL ANSWER:
"""

    # Get response from LLM
    response = llm.invoke(prompt)

    

    # Display formatted output
    print("\n===================================")
    print("📄 CV Assistant Response")
    print("===================================\n")
    print(response.content)
    print("\n")