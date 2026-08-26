"""Streamlit UI for Chat With Your PDF. This is the thing you deploy and show off.

    streamlit run app.py

Upload one or more PDFs, ask a question, and get an answer with page citations.
The sidebar shows exactly which chunks were retrieved and how strongly they
matched -- so a viewer can SEE the retrieval working, not just trust it. That
transparency is a deliberate portfolio touch: it says "I understand what's
happening under the hood."

By default it runs fully offline (the hashing embedder + extractive answers), so
the deployed demo costs nothing and needs no API key. Tick "Use a real LLM" and
provide a key to get fluent answers from a live model.
"""

from __future__ import annotations

import os
import tempfile

import streamlit as st

from pdf_chat.pipeline import Rag

st.set_page_config(page_title="Chat With Your PDF", page_icon="📄", layout="wide")

st.title("📄 Chat With Your PDF")
st.caption(
    "A retrieval-augmented-generation demo. Upload PDFs, ask questions, get answers "
    "with citations to the source page. Runs offline by default."
)


# --- sidebar: settings --------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=8, value=4)
    chunk_size = st.slider("Chunk size (characters)", 200, 1200, 500, step=100)
    overlap = st.slider("Chunk overlap", 0, 300, 100, step=20)
    use_real = st.checkbox("Use a real LLM (needs an API key)", value=False)
    model = st.text_input("Model (LiteLLM)", value="gemini/gemini-1.5-flash") if use_real else None
    if use_real:
        key = st.text_input("API key", type="password",
                            help="Stored only in this session. On Spaces, use repo Secrets instead.")
        if key:
            # LiteLLM reads provider keys from the environment.
            os.environ.setdefault("GEMINI_API_KEY", key)
            os.environ.setdefault("GOOGLE_API_KEY", key)


# --- ingest -------------------------------------------------------------------
uploads = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

# Rebuild the index whenever the uploaded files or the chunk settings change.
signature = (
    tuple(sorted(f.name for f in uploads)) if uploads else (),
    chunk_size,
    overlap,
)
if uploads and st.session_state.get("signature") != signature:
    with st.spinner("Reading, chunking and embedding your PDFs..."):
        paths = []
        tmpdir = tempfile.mkdtemp()
        for f in uploads:
            p = os.path.join(tmpdir, f.name)
            with open(p, "wb") as out:
                out.write(f.getbuffer())
            paths.append(p)
        rag = Rag(chunk_size=chunk_size, overlap=overlap)
        n = rag.ingest(paths)
        st.session_state.rag = rag
        st.session_state.signature = signature
        st.session_state.n_chunks = n
    st.success(f"Indexed {n} chunks from {len(uploads)} file(s).")

if "rag" not in st.session_state:
    st.info("Upload a PDF to begin. No file handy? Run `python generate_data.py` "
            "and upload one from the `data/` folder.")
    st.stop()


# --- ask ----------------------------------------------------------------------
question = st.text_input("Ask a question about your PDF(s):",
                        placeholder="e.g. How many vacation days do employees get?")

if question:
    rag: Rag = st.session_state.rag
    with st.spinner("Retrieving and answering..."):
        answer = rag.ask(question, k=k, offline=not use_real, model=model)

    if answer.refused:
        st.warning(answer.text)
        st.caption(f"Best match scored only {answer.top_score:.3f} -- below the "
                  "confidence threshold, so the app refused rather than guess.")
    else:
        st.markdown("### Answer")
        st.write(answer.text)
        if answer.citations():
            st.markdown("**Sources:** " + "  ".join(f"`{c}`" for c in answer.citations()))

    # The transparency panel: show what retrieval actually pulled.
    with st.sidebar:
        st.header("Retrieved chunks")
        retrieved = rag.retriever.retrieve(question, k=k)
        for i, item in enumerate(retrieved, start=1):
            with st.expander(f"{i}. {item.chunk.citation()}  ·  score {item.score:.3f}"):
                st.write(item.chunk.text)
