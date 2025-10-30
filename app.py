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
