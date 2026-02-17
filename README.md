# 🩺 Health Information Assistant (CDC RAG)

This project is a **Streamlit-based health question answering application** that uses:
- A **ChromaDB vector database** built from CDC health data
- A **Groq-hosted LLM (LLaMA 3.1)** for answering questions
- **Retrieval-Augmented Generation (RAG)** to ensure answers come only from CDC sources

The app allows users to ask health-related questions and receive:
- Clear answers
- Source citations (Source 1, Source 2, etc.)
- A health-themed user interface with a background image

⚠️ **This application provides general health information only and is not medical advice.**

---

## ✨ Features

- Ask health-related questions in natural language  
- Retrieves relevant CDC documents from ChromaDB  
- Uses Groq LLM to generate answers based only on retrieved context  
- Displays cited sources for transparency  
- Health-themed UI with background image  
- Optional reranking using sentence-transformers  

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- ChromaDB  
- Groq API (OpenAI-compatible)  
- Sentence Transformers (optional reranker)  
- dotenv  

---

## 📂 Project Structure

```text
.
├── app.py          # Main Streamlit application
├── chroma_db/      # Persistent Chroma vector database (CDC data)
├── .env            # API key file (not committed)
└── README.md       # Project documentation

