import html
from pathlib import Path

import pandas as pd
import streamlit as st
from satn import extract_text, run_satn

st.set_page_config(page_title="SATN Summarizer", page_icon="📝", layout="wide")
st.markdown(f"<style>{Path(__file__).with_name('style.css').read_text()}</style>",
            unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    detail = st.select_slider("Summary length", ["short", "medium", "detailed"], value="medium")
    ratio = st.slider("Salient content kept", 0.1, 0.6, 0.3, 0.05)
    if st.button("Clear chat"):
        st.session_state.history = []
        st.rerun()

if "history" not in st.session_state:
    st.session_state.history = []

if not st.session_state.history:
    st.markdown(
        """<div class="hero"><h1>📝 SATN Summarizer</h1>
        <p>Attach a PDF, DOCX or TXT with the + button below, or paste text, and press Enter.
        Runs fully offline, no API keys.</p></div>""",
        unsafe_allow_html=True,
    )


def render_result(res, text_len):
    if "error" in res:
        st.error(res["error"])
        return
    summary = res["abstractive_summary"]
    c1, c2, c3 = st.columns(3)
    for col, label, val in [(c1, "Sentences", res["n_sentences"]),
                            (c2, "Original words", text_len),
                            (c3, "Summary words", len(summary.split()))]:
        col.markdown(f'<div class="stat"><b>{val}</b>{label}</div>', unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["📄 Summary", "⭐ Key sentences", "🧠 Knowledge"])
    with t1:
        st.caption("Mode: " + res.get("mode", ""))
        st.markdown(f'<div class="card">{html.escape(summary)}</div>', unsafe_allow_html=True)
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


# replay history
for i, m in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.write(m["user"])
    with st.chat_message("assistant"):
        render_result(m["result"], m["n_words"])
        if "error" not in m["result"]:
            st.download_button("Download summary", m["result"]["abstractive_summary"],
                               f"summary_{i}.txt", key=f"dl_{i}")

# Claude-style input bar with the attach (+) button inside, on the left
prompt = st.chat_input("Attach a file or paste text to summarize...",
                       accept_file=True, file_type=["pdf", "docx", "txt", "md"])

if prompt:
    files = prompt.files if hasattr(prompt, "files") else []
    typed = (prompt.text if hasattr(prompt, "text") else str(prompt)) or ""
    label = typed[:200] + ("..." if len(typed) > 200 else "")
    if files:
        label = f"📎 {files[0].name}" + (f"\n\n{label}" if label else "")
        text = extract_text(files[0].getvalue(), files[0].name)
    else:
        text = typed.strip()

    with st.chat_message("user"):
        st.write(label)
    with st.chat_message("assistant"):
        if len(text.split()) < 50:
            result = {"error": "Please attach a file or paste at least ~50 words."}
            render_result(result, 0)
        else:
            with st.spinner("Running SATN (first run downloads models once)..."):
                result = run_satn(text, ratio=ratio, detail=detail)
            render_result(result, len(text.split()))
    st.session_state.history.append({"user": label, "result": result, "n_words": len(text.split())})
