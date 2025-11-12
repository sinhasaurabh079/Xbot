# ──────────────────────────────────────────────────────────────────────
# X‑Chatbot – Streamlit PDF QA
# ──────────────────────────────────────────────────────────────────────
#  • Uses LangChain 0.2+ (core, community, and vectorstore modules)
#  • Uses LangChain‑Groq LLM (llama3‑8b‑8192)
#  • Stores the vector‑store in st.session_state so it survives
#    reruns but is re‑created only when new PDFs are processed
# ──────────────────────────────────────────────────────────────────────

import os
import warnings
import logging

import streamlit as st

# ------------------------------------------------------------------
# 1️⃣  Imports from LangChain – the newest structure (v0.2+)
# ------------------------------------------------------------------
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
# Vector‑store (FAISS) lives in the community package in v0.2
from langchain_community.vectorstores import FAISS

from langchain.chains import RetrievalQA
# ------------------------------------------------------------------
# 2️⃣  Streamlit configuration
# ------------------------------------------------------------------
st.set_page_config(page_title="X‑Chatbot", page_icon="🤖")
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Groq API key – put it in `secrets.toml` (Streamlit Cloud)
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# ------------------------------------------------------------------
# 3️⃣  Working directory for uploads
# ------------------------------------------------------------------
UPLOAD_DIR = "tmp/uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 4️⃣  Page title and state initialisation
# ------------------------------------------------------------------
st.title("Chat With PDF!")
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ------------------------------------------------------------------
# 5️⃣  Show the chat history
# ------------------------------------------------------------------
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# ------------------------------------------------------------------
# 6️⃣  Helper to build a vector‑store from a list of UploadedFile objects
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_vectorstore(pdf_files):
    """
    * Accepts a list of streamlit UploadedFile objects.
    * Saves each PDF to disk.
    * Loads the text via PyPDFLoader.
    * Splits it into chunks.
    * Embeds with HuggingFace all‑MiniLM‑L12‑v2.
    * Stores the embeddings in a FAISS vector‑store.
    """
    docs = []

    # ---- load PDFs ----------------------------------------------------
    for file in pdf_files:
        if not file.name.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(UPLOAD_DIR, file.name.strip())
        with open(file_path, "wb") as fp:
            fp.write(file.read())

        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())

    if not docs:
        return None

    # ---- chunking -----------------------------------------------------
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # ---- embeddings ---------------------------------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2",
    )

    # ---- vector‑store --------------------------------------------------
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# ------------------------------------------------------------------
# 7️⃣  Sidebar – PDF upload & processing button
# ------------------------------------------------------------------
with st.sidebar:
    st.subheader("Upload your PDFs")
    pdf_docs = st.file_uploader(
        "Only PDF files are allowed",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Process"):
        with st.spinner("Processing PDFs…"):
            if not pdf_docs:
                st.error("⚠️ No PDFs uploaded.")
            else:
                vecstore = build_vectorstore(pdf_docs)
                if vecstore is None:
                    st.error("⚠️ No valid PDFs found.")
                else:
                    st.session_state.vectorstore = vecstore
                    st.success("✅ Documents processed successfully!")

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center;">
            <b>🚀 Developed by Gaurav Sinha</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------
# 8️⃣  Chat input – send prompt to LLM + RetrievalQA
# ------------------------------------------------------------------
prompt = st.chat_input("Ask a question…")
if prompt:
    # Show user message immediately
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # If we haven't uploaded PDFs yet, ask user to do so
    if st.session_state.vectorstore is None:
        st.error("⚠️ Please upload and process PDF files first.")
    else:
        # Prompt template for the system message
        system_template = ChatPromptTemplate.from_template(
            """
            You are an expert assistant. Answer the user's question
            directly using information from the documents you have read.
            No small talk. Just the answer.
            Question: {user_prompt}
            """
        )

        # Initialise the Groq LLM
        groq_llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama3-8b-8192",
        )

        # Build RetrievalQA chain – the vector‑store will supply relevant chunks
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=groq_llm,
                chain_type="stuff",                # simple concatenation of retrieved docs
                retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
            )
            result = qa_chain({"query": prompt})
            answer = result["result"]

            # Show assistant answer
            st.chat_message("assistant").markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as exc:
            st.error(f"❌ Error while generating answer: {exc}")

# ------------------------------------------------------------------
# 9️⃣  Footer
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: rgba(0, 0, 0, 0.8);
        color: #fff;
        text-align: center;
        padding: 10px;
        font-size: 14px;
    }
    </style>
    <div class="footer">
        🚀 Developed by <b>Saurabh Sinha</b>
    </div>
    """,
    unsafe_allow_html=True,
)
