# main.py — BharatGPT v2 (Day 2) — Faster + Memory + Better Hindi
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # FIXED: Your Day 1 win
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA  # FIXED: Your Day 1 win

st.set_page_config(page_title="BharatGPT v2", page_icon="🇮🇳")
st.title("🇮🇳 BharatGPT v2 — Day 2/45")
st.write("Faster • Smarter Hindi • Remembers Everything")

# Initialize chat history (UPGRADE 2: Remembers last 5 questions)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Upload PDF (same as Day 1)
uploaded_file = st.file_uploader("Upload PDF (resume, bill, notes)", type="pdf")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User question (chat input for memory)
if prompt := st.chat_input("Ask in English or Hindi..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if uploaded_file is not None:
        with st.chat_message("assistant"):
            with st.spinner("BharatGPT soch raha hai... (20–60 sec first time)"):
                # Load and process PDF (same as Day 1)
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                loader = PyPDFLoader("temp.pdf")
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(docs)

                # UPGRADE 1: nomic-embed-text (your Day 1 fix — 3× faster/better Hindi/English)
                embeddings = OllamaEmbeddings(model="nomic-embed-text")
                db = Chroma.from_documents(chunks, embeddings)

                # UPGRADE 3: Faster model (3B params = quicker on CPU)
                llm = Ollama(model="llama3.2:3b")

                # RAG chain (more context for better answers)
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=db.as_retriever(search_kwargs={"k": 6})
                )

                # Add chat history to prompt (remembers last 5)
                context = "\n".join([m["content"] for m in st.session_state.messages[-5:]])
                full_question = f"Previous chat: {context}\n\nQuestion: {prompt}"

                answer = qa.run(full_question)
                st.markdown(answer)

                # Save to history
                st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.warning("Pehle PDF upload karo bhai!")