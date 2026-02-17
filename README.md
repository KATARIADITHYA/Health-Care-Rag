# Health-Care-RAG
🩺 CDC Healthcare RAG Assistant

A Retrieval-Augmented Generation (RAG) system that provides accurate, citation-based health information using official CDC content. The system retrieves relevant CDC documents, reranks them for precision, and generates grounded responses using a Large Language Model (Groq LLaMA).

⚠️ This system provides general health information only. It is not medical advice.
![UI Screenshot](https://github.com/KATARIADITHYA/Health-Care-Rag/blob/main/Picture1.jpg)



📌 Project Overview

The CDC Healthcare RAG Assistant is designed to:

Retrieve official CDC health content using semantic vector search

Apply metadata filtering for topic-specific results

Rerank retrieved documents using a cross-encoder model

Generate grounded answers using Groq LLaMA models

Provide transparent source citations

Prevent hallucination through strict prompt constraints

This project demonstrates production-grade AI system design with explainability and healthcare-safe grounding.

🏗️ System Architecture

User Query
→ ChromaDB Vector Retrieval
→ Metadata Filtering
→ Cross-Encoder Reranking
→ Prompt Construction
→ Groq LLaMA Generation
→ Citation-based Answer

🛠️ Tech Stack

Vector Database: ChromaDB (Persistent Client)

Embedding Model: all-MiniLM-L6-v2

Reranker: Cross-Encoder (ms-marco-MiniLM-L-6-v2)

LLM Backend: Groq (LLaMA 3.1)

UI Framework: Streamlit

Environment Management: python-dotenv

Language: Python 3.12

📂 Project Structure

├── app.py                 # Streamlit UI

├── chroma_db/             # Persistent ChromaDB storage

├── .env                   # API keys 

├── requirements.txt       # Python dependencies

└── README.md

🚀 Installation & Setup

1️⃣ Clone the repository
git clone <your-repo-url>
cd <project-folder>

2️⃣ Install dependencies
pip install -r requirements.txt


If needed:

pip install streamlit chromadb sentence-transformers openai python-dotenv

3️⃣ Configure Environment Variables

Create a .env file:

GROQ_API_KEY=***


4️⃣ Run the Application
streamlit run app.py

