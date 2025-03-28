import os
import warnings
import logging
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

# Set page configuration
st.set_page_config(page_title="X-Chatbot", page_icon="🤖")
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
# Load environment variables
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Define the directory to store uploaded files
UPLOAD_DIR = "/tmp/uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.title("Chat With PDF!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])


@st.cache_resource
def get_vectorstore(pdf_docs):
    loaders = []
    for pdf in pdf_docs:
        pdf_path = os.path.join(UPLOAD_DIR, pdf.name.strip())
        with open(pdf_path, "wb") as f:
            f.write(pdf.read())
        loaders.append(PyPDFLoader(pdf_path))

    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2"),
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        ),
    ).from_loaders(loaders)

    return index.vectorstore


# Sidebar for file upload
with st.sidebar:
    st.subheader("Upload your PDFs")
    pdf_docs = st.file_uploader(
        "Upload your PDFs and click 'Process'", accept_multiple_files=True
    )

    if st.button("Process"):
        with st.spinner("Processing..."):
            if not pdf_docs:
                st.error("No PDFs uploaded. Please upload files first.")
            else:
                vectorstore = get_vectorstore(pdf_docs)
                if vectorstore is None:
                    st.error("Failed to load documents")
                else:
                    st.success("Documents processed successfully!")

prompt = st.chat_input("Pass your prompt here")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    groq_sys_prompt = ChatPromptTemplate.from_template(
        """You are very smart at everything, you always give the best, 
                                            the most accurate and most precise answers. Answer the following Question: {user_prompt}.
                                            Start the answer directly. No small talk please"""
    )

    model = "llama3-8b-8192"

    groq_chat = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=model,
    )

    try:
        if "vectorstore" not in locals():
            vectorstore = get_vectorstore(pdf_docs)

        chain = RetrievalQA.from_chain_type(
            llm=groq_chat,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
        )

        result = chain({"query": prompt})
        response = result["result"]

        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"Error: {str(e)}")
