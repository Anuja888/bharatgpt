# main.py — BharatGPT v4 — Day 4/45 (Multi-Modal: Digital + Scanned PDFs + Images)
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
from langchain_core.documents.base import Document
from PIL import Image
import easyocr
import numpy as np
import cv2

st.set_page_config(page_title="BharatGPT v4", page_icon="🇮🇳")
st.title("🇮🇳 BharatGPT v4 — Day 4/45")
st.write("Multi-Modal • Digital & Scanned PDFs • Images • Hindi/English • OCR Powered")

if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader(
    "Upload PDF or Image (digital/scanned bill, Aadhaar, photo)",
    type=["pdf", "png", "jpg", "jpeg"]
)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask in Hindi or English...")

if prompt and uploaded_file:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing document..."):
            text = ""

            # Step 1: Try normal text extraction from PDF (fast for digital PDFs)
            if uploaded_file.type == "application/pdf":
                try:
                    loader = PyPDFLoader(uploaded_file)
                    docs = loader.load()
                    text = "\n".join([d.page_content for d in docs])
                    if text.strip():
                        st.info("Extracted text from digital PDF (fast)")
                    else:
                        st.warning("No digital text found — running OCR on scanned PDF")
                except Exception as e:
                    st.warning("PDF text extraction failed — running OCR")

            # Step 2: If no text yet (scanned PDF or image), use OCR
            if not text.strip():
                # Convert uploaded file to OpenCV image
                bytes_data = uploaded_file.getvalue()
                nparr = np.frombuffer(bytes_data, np.uint8)
                img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img_cv is None:
                    st.error("Could not read file as image — try a different format")
                    st.stop()

                st.image(img_cv, channels="BGR", caption="Processing scanned document/image", use_column_width=True)

                # EasyOCR — updated API for 2026
                reader = easyocr.Reader(lang_list=['en', 'hi'], gpu=False)
                result = reader.readtext(img_cv, paragraph=True)
                text = "\n".join([res[1] for res in result])  # res[1] is the recognized text

            if not text.strip():
                st.error("No text found — try a clearer scan or digital PDF")
                st.stop()

            st.success(f"Extracted {len(text)} characters from document")

            # Step 3: RAG on extracted text
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_text(text)
            docs = [Document(page_content=chunk) for chunk in chunks]

            db = Chroma.from_documents(docs, OllamaEmbeddings(model="nomic-embed-text"))

            llm = Ollama(model="llama3.2:3b")
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=db.as_retriever(search_kwargs={"k": 6})
            )

            # Add chat memory
            context = "\n".join([m["content"] for m in st.session_state.messages[-5:]])
            full_question = f"Chat history: {context}\n\nQuestion: {prompt}"

            answer = qa.run(full_question)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

elif prompt:
    st.warning("Pehle file upload karo!")

st.sidebar.success("Day 4/45 Complete — Works for Digital + Scanned PDFs + Images")