import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 1. Load .env (optional but recommended)
load_dotenv()

# 2. Streamlit UI
st.set_page_config(page_title="RAG PDF Chatbot", layout="centered")
st.title("📄🔍 PDF RAG Chatbot using LangChain")

# 3. Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Please set your OpenAI API key in the environment variable 'OPENAI_API_KEY'")
    st.stop()

# 4. Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# 5. Process PDF and Answer Questions
if uploaded_file:
    with st.spinner("Reading and processing PDF..."):
        # Save uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # Load and split document
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        # Vector store with OpenAI embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="./chroma_db")

        # Retrieval + OpenAI LLM
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        # User input
        user_input = st.text_input("Ask a question about the uploaded PDF:")
        if user_input:
            with st.spinner("Generating answer..."):
                result = qa_chain.run(user_input)
                st.success(result)


