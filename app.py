"""Streamlit interface for the medical-versus-finance MoE."""

from html import escape

import matplotlib.pyplot as plt
import streamlit as st

from moe_system import MoESystem


@st.cache_resource
def get_system():
    return MoESystem()


st.set_page_config(page_title="Medical-Finance MoE", layout="wide")
st.title("Medical \u00b7 Finance Mixture of Experts")
system = get_system()
query = st.text_input("Enter your query")

if st.button("Submit") and query.strip():
    result = system.query(query)
    domain = result["domain"]
    badge_color = "#2e8b57" if domain == "medical" else "#1f77b4"
    st.markdown(
        f'<span style="background-color:{badge_color};color:white;'
        f'padding:0.35rem 0.75rem;border-radius:1rem;font-weight:600;">'
        f"{domain.title()}</span>",
        unsafe_allow_html=True,
    )
    st.write(f"Confidence: {result['confidence']:.2%}")
    st.write("Medical score")
    st.progress(result["scores"]["medical"])
    st.write("Finance score")
    st.progress(result["scores"]["finance"])
    st.markdown(
        f'<div style="border:1px solid #d9d9d9;border-radius:0.5rem;'
        f'padding:1rem;margin:1rem 0;">{escape(result["response"])}</div>',
        unsafe_allow_html=True,
    )

    figure, axis = plt.subplots()
    domains = ["medical", "finance"]
    axis.bar(domains, [result["load"][domain] for domain in domains])
    axis.set_xlabel("Domain")
    axis.set_ylabel("Number of queries routed")
    axis.set_title("Expert Load Distribution \u2014 Benchmark Run")
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

    st.subheader("VRAM Breakdown")
    st.table(
        [
            {"Component": "Router", "Size": "86M", "Precision": "FP16", "VRAM": "0.17 GB"},
            {"Component": "Medical", "Size": "2B", "Precision": "FP16", "VRAM": "4.0 GB"},
            {"Component": "Finance", "Size": "3B", "Precision": "FP16", "VRAM": "6.0 GB"},
            {"Component": "Overhead", "Size": "", "Precision": "", "VRAM": "1.5 GB"},
            {"Component": "Total", "Size": "", "Precision": "", "VRAM": "~11.7 GB"},
        ]
    )
