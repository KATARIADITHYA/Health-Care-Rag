# app.py
# Run: streamlit run app.py
# Requirements:
#   pip install streamlit chromadb python-dotenv openai sentence-transformers
#

import os
import time
from typing import List, Dict, Any, Tuple

import streamlit as st
from dotenv import load_dotenv

import chromadb
from openai import OpenAI

# Optional reranker (recommended)
try:
    from sentence_transformers import CrossEncoder
    _HAS_RERANKER = True
except Exception:
    _HAS_RERANKER = False


# ---------------------------
# Config
# ---------------------------
DEFAULT_DB_PATH = "/Users/adithyakatari/Desktop/suchitra/chroma_db"
DEFAULT_TENANT = "default_tenant"
DEFAULT_DATABASE = "default_database"
DEFAULT_COLLECTION = "cdc_diseases"

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


# ---------------------------
# Helpers
# ---------------------------
def build_context(docs: List[str], metas: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        url = meta.get("url", "")
        title = meta.get("disease") or meta.get("topic") or meta.get("title") or ""
        header = f"(Source {i})"
        if title:
            header += f" {title}"
        if url:
            header += f" — {url}"
        blocks.append(f"{header}\n{doc}".strip())
    return "\n\n".join(blocks)


def build_prompt(context: str, question: str) -> str:
    return f"""
You are a health information assistant.
Answer the question using ONLY the CDC context below.
If the answer is not found, say you do not have enough information.

Rules:
- Be concise and factual
- Do NOT add external knowledge
- Use multiple sources if they provide relevant information
- Cite sources as (Source 1), (Source 2), etc.
- Do not invent sources
- This is general information, not medical advice

CDC CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""".strip()


def dedupe_by_url(docs, metas, ids, dists):
    seen = set()
    out_docs, out_metas, out_ids, out_dists = [], [], [], []
    for doc, meta, _id, dist in zip(docs, metas, ids, dists):
        key = meta.get("url") or _id
        if key in seen:
            continue
        seen.add(key)
        out_docs.append(doc)
        out_metas.append(meta)
        out_ids.append(_id)
        out_dists.append(dist)
    return out_docs, out_metas, out_ids, out_dists


def rerank_cross_encoder(query, docs, metas, ids, dists, top_k=5,
                          model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    reranker = CrossEncoder(model_name)
    pairs = [(query, d) for d in docs]
    scores = reranker.predict(pairs).tolist()

    ranked = sorted(
        list(zip(docs, metas, ids, dists, scores)),
        key=lambda x: x[4],
        reverse=True,
    )[:top_k]

    if not ranked:
        return [], [], [], [], []

    r_docs, r_metas, r_ids, r_dists, r_scores = zip(*ranked)
    return list(r_docs), list(r_metas), list(r_ids), list(r_dists), list(r_scores)


def generate_with_groq(prompt: str, model: str, temperature: float = 0.2) -> str:
    client = get_groq_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content


# ---------------------------
# Cached resources
# ---------------------------
@st.cache_resource
def get_groq_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY in .env")
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


@st.cache_resource
def get_chroma_client(db_path: str):
    return chromadb.PersistentClient(
        path=db_path,
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )


@st.cache_resource
def get_collection(db_path: str, name: str):
    client = get_chroma_client(db_path)
    return client.get_collection(name=name)


# ---------------------------
# UI Styling (NO WHITE)
# ---------------------------
st.set_page_config(page_title="Health Answers (CDC RAG)", page_icon="🩺", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
    color: #e5e7eb;
}
.block-container { padding-top: 1rem; }

/* Header card */
.hero {
    padding: 20px;
    border-radius: 14px;
    background: linear-gradient(135deg, #1e293b, #0f172a);
    box-shadow: 0 4px 14px rgba(0,0,0,0.5);
}
.hero h1 { color: #f9fafb; }
.hero p { color: #cbd5f5; }

/* Bubbles */
.bubble-user {
    background: #1e3a8a;
    border-radius: 12px;
    padding: 12px;
    margin: 8px 0;
}
.bubble-assistant {
    background: #1f2937;
    border-radius: 12px;
    padding: 12px;
    margin: 8px 0;
}

/* Source cards */
.source-card {
    background: #020617;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 10px;
}
.source-meta { color: #94a3b8; }

.small-pill {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 999px;
    background: #1e293b;
    color: #e0f2fe;
    font-size: 0.75rem;
    margin-right: 5px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------
# Session state
# ---------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
if "last_debug" not in st.session_state:
    st.session_state.last_debug = {}


# ---------------------------
# Header
# ---------------------------
st.markdown("""
<div class="hero">
  <h1>🩺 Health Answers (CDC RAG)</h1>
  <p>Ask a health question and get an answer based only on your CDC document collection.</p>
</div>
""", unsafe_allow_html=True)

st.warning("General information only — not medical advice.", icon="⚠️")


# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("⚙️ Clinical Settings")

    db_path = st.text_input("Chroma DB Path", value=DEFAULT_DB_PATH)
    collection_name = st.text_input("Collection Name", value=DEFAULT_COLLECTION)

    model = st.selectbox("Groq Model", ["llama-3.1-8b-instant", "llama-3.1-70b-versatile"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2)

    n_results = st.slider("Vector search n_results", 3, 25, 10)
    dedupe = st.checkbox("De-duplicate by URL", True)
    use_rerank = st.checkbox("Use reranker", True, disabled=not _HAS_RERANKER)
    rerank_top_k = st.slider("Answer sources (top_k)", 1, 10, 5)


# ---------------------------
# Main UI
# ---------------------------
left, right = st.columns([2,1])

with left:
    st.subheader("Ask your question")
    question = st.text_area("", placeholder="Example: What should I do if I have signs of dehydration?", height=100)

    col1, col2 = st.columns(2)
    with col1:
        ask = st.button("🔎 Get Answer", type="primary", use_container_width=True)
    with col2:
        clear = st.button("🧹 Clear Chat", use_container_width=True)

with right:
    st.info("• CDC-only answers\n• With citations\n• Debug view available", icon="💡")


if clear:
    st.session_state.chat = []
    st.session_state.last_sources = []
    st.session_state.last_debug = {}
    st.rerun()


# ---------------------------
# Q&A
# ---------------------------
if ask:
    q = question.strip()
    if not q:
        st.error("Please enter a question.")
        st.stop()

    st.session_state.chat.append({"role": "user", "content": q})

    collection = get_collection(db_path, collection_name)

    results = collection.query(
        query_texts=[q],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    if dedupe:
        docs, metas, _, dists = dedupe_by_url(docs, metas, list(range(len(docs))), dists)

    if use_rerank and _HAS_RERANKER:
        docs, metas, _, dists, _ = rerank_cross_encoder(q, docs, metas, list(range(len(docs))), dists, rerank_top_k)
    else:
        docs = docs[:rerank_top_k]
        metas = metas[:rerank_top_k]

    context = build_context(docs, metas)
    prompt = build_prompt(context, q)

    answer = generate_with_groq(prompt, model, temperature)

    st.session_state.chat.append({"role": "assistant", "content": answer})
    st.session_state.last_sources = metas
    st.rerun()


# ---------------------------
# Render chat
# ---------------------------
st.subheader("Conversation")
for msg in st.session_state.chat:
    if msg["role"] == "user":
        st.markdown(f"<div class='bubble-user'><b>You</b><br>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bubble-assistant'><b>Assistant</b><br>{msg['content']}</div>", unsafe_allow_html=True)


# ---------------------------
# Sources
# ---------------------------
if st.session_state.last_sources:
    st.subheader("Evidence (CDC Sources)")
    for i, meta in enumerate(st.session_state.last_sources, start=1):
        title = meta.get("disease") or meta.get("topic") or "Source"
        url = meta.get("url","")
        st.markdown(f"""
<div class="source-card">
<b>Source {i}: {title}</b><br>
<span class="source-meta">{url}</span>
</div>
""", unsafe_allow_html=True)
