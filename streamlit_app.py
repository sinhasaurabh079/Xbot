import os
import warnings
import logging
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter 

from langchain_community.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

# Streamlit setup
st.set_page_config(page_title="X-Chatbot", page_icon="🤖")
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# API Key (add this in Streamlit Cloud secrets)
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Directory setup
UPLOAD_DIR = "tmp/uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.title("Chat With PDF!")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Display chat history
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])


@st.cache_resource
def get_vectorstore(pdf_docs):
    loaders = []
    for pdf in pdf_docs:
        if not pdf.name.endswith(".pdf"):
            continue  # skip non-pdf files
        pdf_path = os.path.join(UPLOAD_DIR, pdf.name.strip())
        with open(pdf_path, "wb") as f:
            f.write(pdf.read())
        loaders.append(PyPDFLoader(pdf_path))

    if not loaders:
        return None

    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name="models/all-MiniLM-L12-v2"),
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        ),
    ).from_loaders(loaders)

    return index.vectorstore


# Sidebar for PDF upload
with st.sidebar:
    st.subheader("Upload your PDFs")
    pdf_docs = st.file_uploader(
        "Only PDF files are allowed", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Process"):
        with st.spinner("Processing PDFs..."):
            if not pdf_docs:
                st.error("No PDFs uploaded.")
            else:
                vectorstore = get_vectorstore(pdf_docs)
                if vectorstore is None:
                    st.error("No valid PDFs found.")
                else:
                    st.session_state.vectorstore = vectorstore
                    st.success("Documents processed successfully!")

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center;">
            <b>🚀 Developed by Gaurav Sinha</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Chat input
prompt = st.chat_input("Pass your prompt here")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if st.session_state.vectorstore is None:
        st.error("Please upload and process PDF files first.")
    else:
        # Prompt template
        groq_sys_prompt = ChatPromptTemplate.from_template(
            """You are very smart at everything, you always give the best, 
            the most accurate and most precise answers. Answer the following Question: {user_prompt}.
            Start the answer directly. No small talk please"""
        )

        # LLM initialization
        groq_chat = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama3-8b-8192",
        )

        # Retrieval QA
        try:
            chain = RetrievalQA.from_chain_type(
                llm=groq_chat,
                chain_type="stuff",
                retriever=st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": 3}
                ),
                return_source_documents=True,
            )
            result = chain({"query": prompt})
            response = result["result"]

            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: rgba(0, 0, 0, 0.8);
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 14px;
    }
    </style>
    <div class="footer">
        🚀 Developed by <b>Saurabh Sinha</b>
    </div>
    """,
    unsafe_allow_html=True,
)
