# Chat With Your PDF

Upload PDFs, ask questions in plain English, get answers **with citations to the
source page**, and an honest "I don't know" when the documents don't cover it.
This is the classic Retrieval-Augmented Generation (RAG) pipeline, written so you
can read every line of it.

**Problem:** A language model on its own can't answer questions about *your*
documents, and if you force it to, it makes things up. RAG fixes that: find the
relevant passages first, then answer using only those, and show your work.

**Skills demonstrated:** RAG pipeline design, PDF text extraction, chunking with
overlap, embeddings, cosine similarity / vector search, retrieval, prompt
grounding, citations, hallucination guardrails, a Streamlit UI, pytest, packaging.

**Tech stack:** Python, numpy, pypdf, Streamlit, pytest. Optional upgrades:
LiteLLM (real LLM answers), sentence-transformers (neural embeddings), Chroma
(a real vector DB).

## The pipeline

```
 load  ──▶  chunk  ──▶  embed  ──▶  store  ──▶  retrieve  ──▶  generate
 PDF→text   small       text→       keep all    top-k        stuff chunks
 (per page) overlapping  vector      vectors     closest      into a grounded
            passages                             chunks       prompt, cite pages
```

Each arrow is one small module in `pdf_chat/`: `loading.py`, `chunking.py`,
`embedding.py`, `vectorstore.py`, `retrieval.py`, `generate.py`, tied together by
`pipeline.py`.

## Offline by default

The whole thing runs with **no API key and no downloads**. The default embedder
is a TF-IDF hashing vectoriser in plain numpy (a real, classic *lexical* search),
and the default answer is built by quoting the retrieved sentences. That means
every test and demo works on a bare laptop, and it never hallucinates, because it
only ever quotes your documents.

Flip two switches to get the "real" stack:
- `--real` (CLI) or the checkbox (app) gives answers from a live LLM via LiteLLM.
- `pip install sentence-transformers` and use the neural embedder to match by
  *meaning* instead of by shared words (so "biggest" finds "largest").

## Run locally (Windows PowerShell)

```powershell
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python generate_data.py                 # writes 3 sample PDFs into data\

# Command line: index once, then ask as many times as you like
python -m pdf_chat ingest data\remote_work_policy.pdf data\solar_system.pdf data\coffee_guide.pdf --index .index
python -m pdf_chat ask "how many vacation days do I get?" --index .index
python -m pdf_chat ask "how many moons does Jupiter have?" --index .index
python -m pdf_chat ask "what is the capital of France?"   --index .index   # → refuses

# The web app
streamlit run app.py

# The tests
pytest
```

Expected: the vacation question answers "25 days" citing `(remote_work_policy.pdf,
p.2)`; the France question refuses because nothing in the PDFs is relevant.

## Using a real LLM

Get a free Gemini key at <https://aistudio.google.com/app/apikey>, copy
`.env.example` to `.env`, paste your key, then:

```powershell
pip install litellm
python -m pdf_chat ask "summarise the vacation policy" --index .index --real
```

The prompt sent to the model is the same grounded, cite-the-page prompt you can
read in `generate.py`; the LLM just writes it up more fluently.

## What I learned

- RAG is not one big magic call: it's six small, legible steps, and the vector
  DB is "store vectors, return the closest," nothing more.
- The two things that make RAG *trustworthy* are the grounding prompt ("use only
  this context, cite the page") and a guardrail that refuses when retrieval is weak.
- Lexical search (word overlap) is a real technique with a real blind spot,
  synonyms and word collisions, which is exactly the gap neural embeddings fill.

## Layout

```
chat-with-your-pdf/
├── pdf_chat/            the package, one module per pipeline step
│   ├── loading.py       PDF/TXT → pages (keeps page numbers for citations)
│   ├── chunking.py      pages → small overlapping chunks
│   ├── embedding.py     text → vector (offline TF-IDF, or neural if installed)
│   ├── vectorstore.py   store vectors + cosine search (save/load to disk)
│   ├── retrieval.py     question → top-k chunks
│   ├── generate.py      chunks → answer, with citations + the guardrail
│   ├── pipeline.py      Rag(): ingest() and ask()
│   ├── cli.py           `python -m pdf_chat ingest|ask`
│   └── simple_pdf.py    stdlib-only PDF writer (so the samples need no libraries)
├── tests/               25 pytest tests, all offline
├── app.py               the Streamlit UI
├── generate_data.py     writes the 3 sample PDFs
└── requirements.txt
```
