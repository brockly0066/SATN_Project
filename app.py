import html
from pathlib import Path

import pandas as pd
import streamlit as st
from satn import extract_text, run_satn

st.set_page_config(page_title="SATN Summarizer", page_icon="📝", layout="wide")

css = Path(__file__).with_name("style.css").read_text()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

st.markdown(
    """<div class="hero"><h1>📝 SATN Summarizer</h1>
    <p>Semantic-Aware Transformer Network for abstractive summarization and automated
    knowledge extraction. Runs fully offline, no API keys.</p></div>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Settings")
    detail = st.select_slider("Summary length", ["short", "medium", "detailed"], value="medium")
    ratio = st.slider("Salient content kept", 0.1, 0.6, 0.3, 0.05)

left, right = st.columns([1, 1])
with left:
    up = st.file_uploader("Upload PDF / DOCX / TXT", type=["pdf", "docx", "txt", "md"])
with right:
    pasted = st.text_area("...or paste text", height=130)

if st.button("Summarize ✨"):
    text = extract_text(up.getvalue(), up.name) if up else pasted.strip()
    if len(text.split()) < 50:
        st.warning("Please provide at least ~50 words.")
        st.stop()

    with st.spinner("Running SATN (first run downloads models once)..."):
        res = run_satn(text, ratio=ratio, detail=detail)
    if "error" in res:
        st.error(res["error"])
        st.stop()

    summary = res["abstractive_summary"]
    c1, c2, c3 = st.columns(3)
    for col, label, val in [
        (c1, "Sentences", res["n_sentences"]),
        (c2, "Original words", len(text.split())),
        (c3, "Summary words", len(summary.split())),
    ]:
        col.markdown(f'<div class="stat"><b>{val}</b>{label}</div>', unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["📄 Summary", "⭐ Key sentences", "🧠 Knowledge"])
    with t1:
        st.markdown(f'<div class="card">{html.escape(summary)}</div>', unsafe_allow_html=True)
        st.download_button("Download summary", summary, "summary.txt")
    with t2:
        for s in res["salient_sentences"]:
            st.markdown(f"- {s}")
    with t3:
        k = res["knowledge"]
        st.subheader("Key phrases")
        st.markdown("".join(f'<span class="chip">{html.escape(p)}</span>' for p in k["keyphrases"]),
                    unsafe_allow_html=True)
        st.subheader("Named entities")
        st.dataframe(pd.DataFrame(k["entities"]), use_container_width=True)
        st.subheader("Relations")
        st.dataframe(pd.DataFrame(k["relations"], columns=["subject", "relation", "object"]),
                     use_container_width=True)
