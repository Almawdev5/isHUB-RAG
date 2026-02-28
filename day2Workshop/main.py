
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# -----------------------------------------
# 1️⃣ Sample Documents
# -----------------------------------------

documents = [
    # doc1: About Alex
    "My name  is Alex  um 2nd-year IT student at Addis Ababa University interested  in AI, NLP, and full-stack web development.",

    # doc2: Experience
    "I has developed Python automation scripts, worked on REST API integrations, and managed cloud solutions on AWS.",

    # doc3: Career Goal
    "iI aims to become a full-stack developer with AI expertise and build scalable AI-powered applications.",

    # doc4: Achievement
    "I built and deployed a secure full-stack web application using React, FastAPI, and PostgreSQL.",

    # doc5: Project
    "I created a full-stack e-commerce platform with user authentication, product management, and payment integration."
]

docs = [Document(page_content=text) for text in documents]

# -----------------------------------------
# 2️⃣ Load Embedding Model
# -----------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------------------
# 3️⃣ Create Vector Store
# -----------------------------------------

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

print("Vector database created successfully.")

# -----------------------------------------
# 4️⃣ Similarity Search Function
# -----------------------------------------

def search_query(query, top_k=2):
    print(f"\n🔎 Query: {query}")
    print(f"Top-K: {top_k}")

    results = vectorstore.similarity_search(query, k=top_k)

    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(result.page_content)

# -----------------------------------------
# 5️⃣ Run Example Searches
# -----------------------------------------

search_query("Who is Almaw ", top_k=1)
search_query("What is my experience?", top_k=2)
