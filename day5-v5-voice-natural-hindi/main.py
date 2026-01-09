# main.py — BharatGPT v5 — Day 5/45 (Voice Input + Smarter Hindi)
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
from langchain_core.documents.base import Document
import speech_recognition as sr

st.set_page_config(page_title="BharatGPT v5", page_icon="🇮🇳")
st.title("🇮🇳 BharatGPT v5 — Day 5/45")
st.write("Voice Input • Smarter Hindi • Remembers Everything")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Voice input
prompt = None
if st.button("🎤 Speak in Hindi or English"):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Bol rahe ho... (Speak now – 5-10 sec)")
        audio = r.listen(source, timeout=5, phrase_time_limit=10)
    with st.spinner("Sun raha hoon..."):
        try:
            prompt = r.recognize_google(audio, language="hi-IN")  # Hindi priority
            st.success(f"Sunaa: {prompt}")
        except sr.UnknownValueError:
            st.error("Voice samajh nahi aaya – thoda clear bolo")
        except sr.RequestError:
            st.error("Internet issue – Google API se connect nahi ho raha")
        except Exception as e:
            st.error(f"Voice error: {str(e)} – type kar do")

# Fallback: text input
if prompt is None:
    prompt = st.chat_input("Ya type karo yahan...")

if prompt and uploaded_file:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Jawab de raha hoon..."):
            try:
                # Save uploaded PDF temporarily
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Load & split PDF
                loader = PyPDFLoader("temp.pdf")
                docs = loader.load()
                chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

                # Embeddings + vector DB
                db = Chroma.from_documents(chunks, OllamaEmbeddings(model="nomic-embed-text"))

                # LLM + QA chain
                llm = Ollama(model="llama3.2:3b")

                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=db.as_retriever(search_kwargs={"k": 6})
                )

                # Stronger Hindi prompt + memory
                context = "\n".join([m["content"] for m in st.session_state.messages[-5:]])
                full_question = f"""
Previous chat: {context}

Answer only in natural and simple Hindi.  
Do NOT use any English words.  
Use very friendly, clear, and short sentences.  
Take exact information from the PDF.
Question: {prompt}
"""

                answer = qa.run(full_question)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Error: {str(e)} — PDF ya Ollama check karo")

elif prompt:
    st.warning("Pehle PDF upload karo!")

st.sidebar.success("Day 5/45 — Voice + Smarter Hindi Complete")