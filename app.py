import streamlit as st
import cohere
import os
from dotenv import load_dotenv
import json
import numpy as np

# Load environment variables
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PRODUCTS_FILE = os.getenv("DATABASE_FILE")

# Initialize Cohere client
co = cohere.Client(COHERE_API_KEY)

# ---- Streamlit page setup ----
st.set_page_config(page_title="Product Recommender Chatbot", page_icon="💬", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .user-message-container {
        margin: 0 auto;
        width: 50%;
        display: flex;
        justify-content: flex-end;
    }
    .user-bubble {
        background-color: #303030;
        padding: 10px 15px;
        border-radius: 20px;
        margin: 10px 0;
        display: inline-block;
        max-width: 80%;
        word-wrap: break-word;
        align-self: flex-end;
    }
    .assistant-message-container {
        margin: 0 auto;
        width: 50%;
    }
    .assistant-bubble {
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💬 Product Recommender Chatbot")

# ---- Load product catalog ----
with open(PRODUCTS_FILE, "r", encoding="utf-8") as f:
    products = json.load(f)

# Convert product info to plain text for embeddings
def product_to_text(p):
    parts = [
        f"Name: {p.get('name')}",
        f"Category: {p.get('category')}",
        f"Price INR: {p.get('price')}",
        f"RAM: {p.get('ram_gb')}GB" if p.get('ram_gb') else "",
        f"SSD Storage: {p.get('ssd_storage_gb')}GB" if p.get('ssd_storage_gb') else "",
        f"GPU: {p.get('gpu')}" if p.get('gpu') else "",
        f"CPU: {p.get('cpu')}" if p.get('cpu') else "",
        f"Notes: {p.get('notes','')}"
    ]
    return " | ".join([x for x in parts if x])

documents = [product_to_text(p) for p in products]

# ---- Create embeddings ----
doc_embeddings = co.embed(
    texts=documents,
    model="embed-english-v3.0",
    input_type="search_document"
).embeddings
st.write("Current product catalog- ", len(products), "laptops")
st.write("Ask away!")

# ---- Helper functions ----
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

def find_exact_match(query):
    q = query.lower()
    matches = [p for p in products if p["name"].lower() in q]
    return matches

def semantic_search(query, top_k=5, threshold=0.55):
    query_emb = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query"
    ).embeddings[0]
    sims = [cosine_similarity(query_emb, doc) for doc in doc_embeddings]
    idxs = np.argsort(sims)[::-1]
    results = [(products[i], sims[i]) for i in idxs if sims[i] >= threshold]
    if not results:
        results = [(products[i], sims[i]) for i in idxs[:top_k]]
    return results

def recommend_with_context(query, retrieved):
    facts = [
        {
            "name": p["name"],
            "price": p.get("price"),
            "ram_gb": p.get("ram_gb"),
            "ssd_storage_gb": p.get("ssd_storage_gb"),
            "gpu": p.get("gpu"),
            "notes": p.get("notes","")
        }
        for p, _ in retrieved
    ]
    prompt = (
        "You are a product recommendation assistant. Use ONLY the products provided.\n"
        "Return 1–2 product suggestions with a short (2–3 sentence) justification. "
        "If there is only one match, provide a response specific to that product. "
        "If none match well, say so and suggest the nearest alternatives.\n"
        "The value in price is in INR (Indian Rupees)\n\n"
        f"User query: {query}\n\nProducts:\n{json.dumps(facts, ensure_ascii=False)}"
    )
    chat_history = [{"role": m["role"], "message": m["content"]} for m in st.session_state.messages]

    response = co.chat(
        model="command-r-plus-08-2024",
        message=prompt,
        chat_history=chat_history # Pass the chat history for better context
    )
    return response.text

# ---- Session State ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

# ---- Message Sending ----
def send_message():
    user_message = st.session_state.chat_input.strip()
    if not user_message:
        return

    st.session_state.messages.append({"role": "USER", "content": user_message})

    # Exact match
    exact = find_exact_match(user_message)
    if exact:
        retrieved = [(exact[0], 1.0)]
        reply = recommend_with_context(user_message, retrieved)
        st.session_state.messages.append({"role": "CHATBOT", "content": reply})
        st.session_state.chat_input = ""
        return

    # Otherwise do semantic search + reasoning
    candidates = semantic_search(user_message, top_k=5)
    reply = recommend_with_context(user_message, candidates[:3])
    st.session_state.messages.append({"role": "CHATBOT", "content": reply})

    st.session_state.chat_input = ""

# ---- Display Chat ----
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "USER":
            st.markdown(
                f"<div class='user-message-container'><div class='user-bubble'>{msg['content']}</div></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='assistant-message-container'><div class='assistant-bubble'>{msg['content']}</div></div>",
                unsafe_allow_html=True,
            )

# ---- Input Area ----
input_container = st.container()
with input_container:
    st.markdown("""
    <style>
    div[data-testid="stTextInput"] {
        margin: 0 auto;
        width: 50% !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.text_input(
        "Ask about products:",
        key="chat_input",
        on_change=send_message,
        placeholder="e.g. I need a laptop for video editing under 80000",
        label_visibility="collapsed",
        autocomplete="off",
    )
