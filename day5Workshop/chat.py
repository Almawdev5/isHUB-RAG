import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

# -----------------------------
# Initialize Session Storage
# -----------------------------
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# -----------------------------
# Page Title
# -----------------------------
st.title("📄 Document Q&A Assistant")

# -----------------------------
# PDF Upload Section
# -----------------------------
pdf_file = st.file_uploader("Select a PDF document", type=["pdf"])

if pdf_file is not None:
    
    upload_payload = {
        "file": (
            pdf_file.name,
            pdf_file.getvalue(),
            "application/pdf"
        )
    }

    upload_response = requests.post(
        f"{API_URL}/upload",
        files=upload_payload
    )

    if upload_response.status_code == 200:
        st.success(upload_response.json().get("message"))
    else:
        st.error("Upload failed")

# -----------------------------
# Question Input
# -----------------------------
user_question = st.text_input("Enter your question about the document")

if st.button("Get Answer") and user_question:

    ask_response = requests.post(
        f"{API_URL}/ask",
        json={"question": user_question}
    )

    result = ask_response.json().get("answer")

    # Display answer
    st.markdown("### Response")
    st.write(result)

    # Save question and answer in history
    st.session_state.qa_history.insert(0, {
        "q": user_question,
        "a": result
    })

    # Keep only last 10 entries
    st.session_state.qa_history = st.session_state.qa_history[:10]

# -----------------------------
# History Section
# -----------------------------
st.divider()
st.subheader("Previous Questions (Last 10)")

for record in st.session_state.qa_history:
    with st.expander(record["q"]):
        st.write(record["a"])