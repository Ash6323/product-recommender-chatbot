# product_recommender_full.py
import streamlit as st
import cohere
import os
from dotenv import load_dotenv
import json
import numpy as np
import re
from datetime import datetime

# ---------- CACHING HELPERS ----------
@st.cache_resource
def get_cohere_client(api_key):
    return cohere.Client(api_key)


@st.cache_data
def load_products_cached(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def build_documents_cached(products):
    def product_to_text(p):
        parts = [
            f"Name: {p.get('name')}",
            f"Category: {p.get('category')}",
            f"Price in Rupees: {p.get('price_in_rupees')}",
            f"RAM: {p.get('ram_gb')}GB" if p.get('ram_gb') else "",
            f"Storage: {p.get('ssd_storage_gb')}GB" if p.get('ssd_storage_gb') else "",
            f"GPU: {p.get('gpu')}" if p.get('gpu') else "",
            f"CPU: {p.get('cpu')}" if p.get('cpu') else "",
            f"Notes: {p.get('notes','')}"
        ]
        return " | ".join([x for x in parts if x])

    return [product_to_text(p) for p in products]


@st.cache_data(show_spinner="🔄 Embedding product catalog...")
def embed_documents_cached(docs, api_key):
    client = cohere.Client(api_key)
    embeddings = client.embed(
        texts=docs,
        model="embed-english-v3.0",
        input_type="search_document"
    ).embeddings
    return np.array(embeddings)


@st.cache_data(max_entries=500)
def embed_query_cached(query, api_key):
    client = cohere.Client(api_key)
    return client.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query"
    ).embeddings[0]

load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PRODUCTS_FILE = os.getenv("PRODUCTS_FILE", "data/products.json")

if not COHERE_API_KEY:
    raise RuntimeError("COHERE_API_KEY missing. Put it in your .env or environment variables.")

co = get_cohere_client(COHERE_API_KEY)

st.set_page_config(page_title="Product Recommender Chatbot", page_icon="💬", layout="wide")

# ---------- UI CSS (keeps your original look) ----------
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

# show sample uploaded image (developer requested using uploaded file path)
# if you want to show the uploaded screenshot - path from history:
SAMPLE_IMAGE_PATH = "/mnt/data/2e42ff76-4ec9-4f5f-a8be-715ac76835b0.png"
if os.path.exists(SAMPLE_IMAGE_PATH):
    st.image(SAMPLE_IMAGE_PATH, width=280)

# ---------- Load products ----------
products = load_products_cached(PRODUCTS_FILE)

# Build text for each product (used for embedding & display)
def product_to_text(p):
    parts = [
        f"Name: {p.get('name')}",
        f"Category: {p.get('category')}",
        f"Price in Rupees: {p.get('price_in_rupees')}",
        f"RAM: {p.get('ram_gb')}GB" if p.get('ram_gb') else "",
        f"Storage: {p.get('ssd_storage_gb')}GB" if p.get('ssd_storage_gb') else "",
        f"GPU: {p.get('gpu')}" if p.get('gpu') else "",
        f"CPU: {p.get('cpu')}" if p.get('cpu') else "",
        f"Notes: {p.get('notes','')}"
    ]
    return " | ".join([x for x in parts if x])

documents = build_documents_cached(products)

# ---------- Create embeddings once ----------
doc_embeddings = embed_documents_cached(documents, COHERE_API_KEY)
st.success(f"Catalog ready — {len(products)} products loaded")

# ---------- Helpers ----------
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))

# Extract budget from user query (supports INR variants: 1 lakh, 100000, 80k, 80K)
def extract_budget_in_inr(query: str):
    q = query.lower().replace(",", "").strip()
    q = q.replace("lack", "lakh")

    # lakh / lac
    m_lakh = re.search(r"(\d+(?:\.\d+)?)\s*(lakh|lac)", q)
    if m_lakh:
        return int(float(m_lakh.group(1)) * 100000)

    # k (thousand)
    m_k = re.search(r"(\d+(?:\.\d+)?)\s*k\b", q)
    if m_k:
        return int(float(m_k.group(1)) * 1000)

    # explicit large numbers only
    m_big = re.search(r"\b(\d{5,7})\b", q)
    if m_big:
        return int(m_big.group(1))

    return None

# filter products by budget
def filter_by_budget(products_list, budget_in_inr):
    if budget_in_inr is None:
        return products_list, False
    filtered = [p for p in products_list if p.get("price_in_rupees") is not None and p["price_in_rupees"] <= budget_in_inr]
    # if filtered empty, return empty and a flag
    return filtered, len(filtered) == 0

# exact name match (case-insensitive)
def find_exact_match(query: str):
    q = query.lower().strip()
    matches = [p for p in products if p["name"].lower() == q or p["name"].lower() in q]
    return matches

# detect if query mentions a product name (first/last or full)
def find_product_indices_in_text(text: str, products_list):
    q = text.lower()
    hits = []
    for i, p in enumerate(products_list):
        name = p["name"].lower()
        parts = name.split()
        if name in q or any(part in q for part in parts):
            hits.append(i)
    # dedupe preserve order
    out = []
    seen = set()
    for h in hits:
        if h not in seen:
            out.append(h)
            seen.add(h)
    return out

# semantic search over a subset of indices
def semantic_search_over_indices(query: str, indices, top_k=5, threshold=0.55):
    if not indices:
        return []
    # query_emb = co.embed(texts=[query], model="embed-english-v3.0", input_type="search_query").embeddings[0]
    query_emb = embed_query_cached(query, COHERE_API_KEY)
    sims = []
    for i in indices:
        sims.append((i, cosine_similarity(query_emb, doc_embeddings[i])))
    sims.sort(key=lambda x: x[1], reverse=True)
    results = [(products[i], score) for i, score in sims if score >= threshold]
    if not results:
        results = [(products[i], score) for i, score in sims[:top_k]]
    return results

# build prompt and call LLM for final recommendation (1-2 items, short justification)
def recommend_with_context(query: str, candidate_products):
    facts = [{
        "id": p.get("id"),
        "name": p.get("name"),
        "price_in_rupees": p.get("price_in_rupees"),
        "ram_gb": p.get("ram_gb"),
        "ssd_storage_gb": p.get("ssd_storage_gb"),
        "gpu": p.get("gpu"),
        "notes": p.get("notes","")
    } for p, _ in candidate_products]

    system_prompt = (
        "You are a product recommendation assistant. Use ONLY the products given in 'Products'. "
        "Always respect user intent from entire conversation history. "
        "If the user earlier said 'gaming', continue recommending gaming laptops unless the user changes requirement. "
        "Return at most 1–2 products with short justification."
    )

    # Chat history except system prompt
    chat_history = [
        {"role": m["role"], "message": m["content"]}
        for m in st.session_state.messages
    ]

    # NEW user message = facts + query
    final_user_message = f"{query}\n\nProducts: {json.dumps(facts, ensure_ascii=False)}"

    resp = co.chat(
        model="command-r-plus-08-2024",
        preamble=system_prompt,
        message=final_user_message,
        chat_history=chat_history
    )

    return resp.text


# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

if "loading" not in st.session_state:
    st.session_state.loading = False

# ---------- send_message logic (main) ----------
def send_message():
    st.session_state.loading = True

    user_message = st.session_state.chat_input.strip()
    if not user_message:
        st.session_state.loading = False
        return

    # append user message to history (use role names Cohere accepts)
    st.session_state.messages.append({"role": "USER", "content": user_message})

    # 1) exact match first (title exact or contained)
    exact = find_exact_match(user_message)
    if exact:
        retrieved = [(exact[0], 1.0)]
        answer = recommend_with_context(user_message, retrieved)
        st.session_state.messages.append({"role": "CHATBOT", "content": answer})
        st.session_state.chat_input = ""
        st.session_state.loading = False
        return

    # 2) extract budget if any
    budget = extract_budget_in_inr(user_message)  # integer INR or None

    # 3) filter products by budget if present
    filtered_products, filtered_empty = filter_by_budget(products, budget)

    # if budget present but no products match, we will later relax and tell user
    # 4) build candidate indices to search (if budget used, search only within filtered_products)
    if budget is None:
        candidate_indices = list(range(len(products)))
    else:
        # map filtered_products back to original indices
        candidate_indices = [i for i, p in enumerate(products) if p in filtered_products]

    # 5) Always force-include any product explicitly named in the user query
    name_hits = find_product_indices_in_text(user_message, products)
    # ensure name_hits are included in candidate_indices
    for idx in name_hits:
        if idx not in candidate_indices:
            candidate_indices.insert(0, idx)

    # 6) if there are no candidate indices (budget too tight), relax: use full catalog but remember to inform LLM
    relaxed = False
    if not candidate_indices:
        relaxed = True
        candidate_indices = list(range(len(products)))

    # 7) semantic search over candidate indices
    candidates = semantic_search_over_indices(user_message, candidate_indices, top_k=6, threshold=0.55)

    # 8) If budget was provided and no filtered products matched, prepare a helpful message:
    if budget is not None and filtered_empty and not candidates:
        # no items under budget; let LLM consider nearest items (use full list)
        candidates = semantic_search_over_indices(user_message, list(range(len(products))), top_k=6, threshold=0.0)
        # Ask LLM to explain that nothing fits and provide alternatives
        answer = recommend_with_context(user_message + f" (user budget={budget} INR)", candidates[:4])
        st.session_state.messages.append({"role": "CHATBOT", "content": answer})
        st.session_state.chat_input = ""
        st.session_state.loading = False
        return

    # 9) If candidates empty (rare), fallback to top semantic from full catalog
    if not candidates:
        candidates = semantic_search_over_indices(user_message, list(range(len(products))), top_k=6, threshold=0.0)

    # 10) take up to top N candidates and call LLM to produce 1-2 final suggestions
    final_candidates = candidates[:4]
    answer = recommend_with_context(user_message + (f" (budget={budget} INR)" if budget else ""), final_candidates)
    st.session_state.messages.append({"role": "CHATBOT", "content": answer})

    st.session_state.chat_input = ""
    st.session_state.loading = False

if st.session_state.loading:
    st.markdown(
        """
        <div style="width:50%; margin: 0 auto; text-align:center; padding:10px;">
            🤖 Thinking...
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------- Display chat history ----------
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "USER":
            st.markdown(f"<div class='user-message-container'><div class='user-bubble'>{msg['content']}</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-message-container'><div class='assistant-bubble'>{msg['content']}</div></div>", unsafe_allow_html=True)

# ---------- Input area ----------
input_container = st.container()
with input_container:
    left, center, right = st.columns([1, 2, 1])

    with center:
        col1, col2 = st.columns([9, 1])

        with col1:
            st.text_input(
                "Ask about products:",
                key="chat_input",
                placeholder=f"e.g. I need a laptop for video editing under 80000 or Tell me about {products[0]['name']}",
                label_visibility="collapsed",
                autocomplete="off",
                on_change=send_message
            )

