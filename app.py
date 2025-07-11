import streamlit as st
from utils.pdf_utils import extract_text_from_pdf
from utils.vector_store import load_vectorstore_from_pdf
from utils.llm_utils import get_answer

st.set_page_config(page_title="📄 PDF Chatbot", layout="centered")
st.title("📚 Ask Your PDF Anything (Powered by OpenAI + FAISS)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "document_name" not in st.session_state:
    st.session_state.document_name = ""

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file:
    st.session_state.document_name = uploaded_file.name
    with st.spinner("Reading and indexing the PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
        vectorstore = load_vectorstore_from_pdf(pdf_text)
        st.session_state.vectorstore = vectorstore
    st.success(f"✅ {uploaded_file.name} uploaded and ready!")

if st.session_state.vectorstore:
    st.markdown(f"**Chatting with:** `{st.session_state.document_name}`")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input("Ask a question about the PDF...")

    if user_question:
        st.chat_message("user").markdown(user_question)
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = get_answer(
                    question=user_question,
                    vectorstore=st.session_state.vectorstore,
                    chat_history=st.session_state.chat_history
                )
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
