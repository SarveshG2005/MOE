"""Streamlit interface for the Medical-Finance MoE — HuggingFace Spaces version.
   Router runs live. Expert answers served from pre-generated cache.
"""

import json
import os
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
LABELS      = {0: "medical", 1: "finance"}
ROUTER_PATH = "router/final"
CACHE_PATH  = "answer_cache.json"

# ── Load router (CPU, 86M — fits HF free tier) ────────────────────────────────
@st.cache_resource
def load_router():
    tokenizer = AutoTokenizer.from_pretrained(ROUTER_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        ROUTER_PATH, num_labels=2,
    )
    model.eval()
    return tokenizer, model

# ── Load cache ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_cache():
    with open(CACHE_PATH, "r") as f:
        return json.load(f)

def route(query, tokenizer, model):
    inputs = tokenizer(
        query, return_tensors="pt", truncation=True, max_length=512,
    )
    with torch.inference_mode():
        logits = model(**inputs).logits
    probs     = F.softmax(logits, dim=-1)[0]
    label_id  = torch.argmax(probs).item()
    domain    = LABELS[label_id]
    scores    = {LABELS[i]: round(float(probs[i]), 4) for i in range(len(LABELS))}
    return domain, round(float(probs[label_id]), 4), scores

def get_answer(query, domain, cache):
    # Exact match first
    if query in cache:
        return cache[query]["answer"]
    # Fallback — find any cached answer for the domain
    for v in cache.values():
        if v["domain"] == domain:
            return v["answer"] + "\n\n*(Sample answer — live inference not available on free CPU)*"
    return "No cached answer available for this domain."

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Medical-Finance MoE", layout="wide")
st.title("Medical · Finance Mixture of Experts")
st.caption("Router runs live on CPU · Expert answers pre-generated on T4 GPU (Kaggle)")

router_tokenizer, router_model = load_router()
cache = load_cache()

query = st.text_input("Enter your query")

if st.button("Submit") and query.strip():
    domain, confidence, scores = route(query, router_tokenizer, router_model)
    answer = get_answer(query, domain, cache)

    badge_color = "#2e8b57" if domain == "medical" else "#1f77b4"
    st.markdown(
        f'<span style="background-color:{badge_color};color:white;'
        f'padding:0.35rem 0.75rem;border-radius:1rem;font-weight:600;">'
        f"{domain.title()}</span>",
        unsafe_allow_html=True,
    )
    st.write(f"Confidence: {confidence:.2%}")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Medical score")
        st.progress(scores["medical"])
    with col2:
        st.write("Finance score")
        st.progress(scores["finance"])

    st.markdown(
        f'<div style="border:1px solid #d9d9d9;border-radius:0.5rem;'
        f'padding:1rem;margin:1rem 0;">{escape(answer)}</div>',
        unsafe_allow_html=True,
    )

    st.subheader("VRAM Breakdown (4-bit quantization, Kaggle T4)")
    st.table([
        {"Component": "Router",   "Size": "86M", "Precision": "FP16",  "VRAM": "0.38 GB"},
        {"Component": "Medical",  "Size": "2B",  "Precision": "4-bit", "VRAM": "~2.5 GB"},
        {"Component": "Finance",  "Size": "3B",  "Precision": "4-bit", "VRAM": "~2.0 GB"},
        {"Component": "Overhead", "Size": "",    "Precision": "",      "VRAM": "~0.5 GB"},
        {"Component": "Total",    "Size": "",    "Precision": "",      "VRAM": "~4.92 GB"},
    ])