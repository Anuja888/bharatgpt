import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
import speech_recognition as sr
import torch

st.set_page_config(page_title="BharatGPT v3", page_icon="🇮🇳")
st.title("🇮🇳 BharatGPT v3 — Day 3/45")
st.write("GPU Powered • Speak Hindi/English • Voice Input • Memory")

if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader("Upload PDF (resume, notes, bill)", type="pdf")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = None
if st.button("🎤 Speak your question (Hindi/English)"):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Boliye... (Speak now)")
        audio = r.listen(source, timeout=5, phrase_time_limit=10)
    with st.spinner("Processing voice..."):
        try:
            prompt = r.recognize_google(audio, language="hi-IN")
            st.success(f"Sunaa: {prompt}")
        except:
            try:
                prompt = r.recognize_google(audio, language="en-IN")
                st.success(f"Sunaa (English): {prompt}")
            except:
                st.error("Voice nahi samajh aaya — type karo")
                prompt = None

if not prompt:
    prompt = st.chat_input("Or type here...")

if prompt and uploaded_file:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Jawab de raha hoon..."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            docs = PyPDFLoader("temp.pdf").load()
            chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            db = DocArrayInMemorySearch.from_documents(chunks, embeddings)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            st.info(f"Running on: **{device.upper()}**")

            llm = Ollama(model="llama3.2:3b")
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever(search_kwargs={"k": 6}))

            context = "\n".join([m["content"] for m in st.session_state.messages[-5:]])
            full_question = f"Previous: {context}\n\nQuestion: {prompt}"

            answer = qa.run(full_question)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

elif prompt:
    st.warning("Pehle PDF upload karo!")

if torch.cuda.is_available():
    st.success(f"🚀 GPU ON: {torch.cuda.get_device_name(0)}")
else:
    st.info("CPU mode — still solid")