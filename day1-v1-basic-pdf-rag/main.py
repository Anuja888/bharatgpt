import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA

st.set_page_config(page_title="Day 1 — PDF Guru", page_icon="fire")
st.title("fire Day 1/45 — Private PDF Guru (100% Local)")

uploaded_file = st.file_uploader("Upload any PDF (resume, notes, bill)", type="pdf")
question = st.text_input("Ask anything in English or Hindi")

if uploaded_file and question:
    with st.spinner("Reading your PDF... (first run 30–90 sec, then fast)"):
        # Save uploaded file
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load and split
        docs = PyPDFLoader("temp.pdf").load()
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

        # THIS LINE MAKES IT WORK FAST & RELIABLE
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # Create vector DB
        db = Chroma.from_documents(chunks, embeddings)

        # LLM + QA
        llm = Ollama(model="llama3.2")
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())

        # Get answer
        answer = qa.run(question)
        st.success("Answer:")
        st.write(answer)