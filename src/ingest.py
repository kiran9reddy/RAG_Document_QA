import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

def load_pdf_text(pdf_path):
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            texts.append(page.extract_text())
    return "\n".join(texts)

def split_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

def create_embeddings(texts, persist_directory="vectorstore"):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_texts(texts, embeddings, persist_directory=persist_directory)
    vectordb.persist()
    return vectordb

def ingest_pdfs(pdf_folder="data/", persist_directory="vectorstore"):
    all_texts = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            text = load_pdf_text(os.path.join(pdf_folder, file))
            chunks = split_text(text)
            all_texts.extend(chunks)
    vectordb = create_embeddings(all_texts, persist_directory=persist_directory)
    return vectordb
