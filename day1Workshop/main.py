import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# -----------------------------
# Step 1: Load Environment Variables
# -----------------------------
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

# -----------------------------
# Step 2: Your Documents
# -----------------------------
documents = [
    {
        "title": "About Almaw Tadele",
        "content": (
            "I am Almaw Tadele, a 2nd-year IT student at Addis Ababa University. "
            "I am learning Full-Stack development, including frontend with React and Next.js, "
            "and backend basics. I am interested in AI, chatbots, and RAG systems, which is why I am in isHUB learning this course."
        )
    },
    {
        "title": "Education and Skills",
        "content": (
            "I study Information Technology at Addis Ababa University. "
            "I have skills in HTML, CSS, JavaScript, Python, and database basics. "
            "I am improving my knowledge in AI, natural language processing, and web development to support my career and help my family."
        )
    }
]

# -----------------------------
# Step 3: Simple Retriever
# -----------------------------
def retrieve_doc(query):
    for doc in documents:
        if query.lower() in doc["title"].lower():
            return doc["content"]
    return "No document found."

# -----------------------------
# Step 4: Initialize Groq LLM
# -----------------------------
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0,
    groq_api_key=groq_api_key
)

# -----------------------------
# Step 5: Prompt Engineering Demo
# -----------------------------
print("=== Prompt Engineering Demo ===\n")

basic_question = "Explain Artificial Intelligence."
print("Basic Prompt Output:\n")
print(llm.invoke(basic_question).content)

improved_prompt = """
Explain Artificial Intelligence to a high school student.
Use simple language and 2 real-life examples.
"""
print("\nImproved Prompt Output:\n")
print(llm.invoke(improved_prompt).content)

# -----------------------------
# Step 6: RAG Demo
# -----------------------------
print("\n=== RAG Demo: Almaw Tadele ===\n")

user_question = "Who is Almaw Tadele?"

# WITHOUT retrieval
print("Answer WITHOUT retrieval:\n")
print(llm.invoke(user_question).content)

# WITH retrieval
context = retrieve_doc("About Almaw Tadele")

rag_prompt = f"""
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{user_question}
"""
print("\nAnswer WITH retrieval (RAG):\n")
print(llm.invoke(rag_prompt).content)

# -----------------------------
# Step 7: Workflow Visualization
# -----------------------------
print("\n=== RAG Workflow ===\n")
print("""
User Question
      |
Retriever (search documents list)
      |
Retrieved Context
      |
Context + Question → Prompt
      |
Groq LLM
      |
Final Answer
""")