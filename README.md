# RAG Document QA 🔍

A Retrieval-Augmented Generation (RAG) project for **question-answering over documents**.  
Upload PDFs, generate embeddings, and query them with a large language model (LLM) to get accurate answers.  

---

## **Project Overview**

This project demonstrates:

- **Document ingestion:** Extract text from PDFs and split into chunks.  
- **Vector database:** Store embeddings using Chroma + Sentence Transformers.  
- **RAG-based QA:** Combine retrieval of relevant document chunks with LLM for natural language answers.  
- **Flexible:** Run locally or on Google Colab.  

---

## **Folder Structure**

RAG_Document_QA/
├── data/ # Store your PDF documents here
├── notebooks/
│ └── rag_demo.ipynb # Demo notebook for Colab
├── src/
│ ├── ingest.py # PDF ingestion and embedding creation
│ ├── model_utils.py # Load LLM and build retrieval chain
│ └── qa_system.py # Query and get answers
├── README.md # Project overview
└── requirements.txt # Dependencies


**Usage**
1. Ingest PDFs
from src.ingest import ingest_pdfs
vectordb = ingest_pdfs(pdf_folder="data/", persist_directory="vectorstore")

2. Query the documents
from src.qa_system import answer_question
question = "What is the main topic of the PDF?"
answer = answer_question(question)
print(answer)

3. Run on Google Colab

Clone repo in Colab:

!git clone https://github.com/<your-username>/RAG_Document_QA.git
%cd RAG_Document_QA

