"""pdf_chat -- a "Chat With Your PDF" RAG engine you can read end to end.

RAG stands for Retrieval-Augmented Generation. In one sentence: instead of
hoping a language model already knows the answer, we FIND the relevant passages
in your own documents first, then hand those passages to the model and ask it to
answer using only them -- with a citation back to the page it came from.

The pipeline, in the order the data flows:

    load  ->  chunk  ->  embed  ->  store  ->  retrieve  ->  generate

    loading.py      turn a PDF into text, one page at a time (keeps page numbers)
    chunking.py     cut each page into small overlapping passages
    embedding.py    turn a passage into a vector (a list of numbers)
    vectorstore.py  keep all the vectors and find the closest ones to a query
    retrieval.py    embed the question, fetch the top-k closest passages
    generate.py     stuff those passages into a prompt and produce an answer
    pipeline.py     wire all of the above into one Rag object

Two things make this codebase unusual, both on purpose:

  * It runs with NO API key and NO downloads. The default embedder is a small
    keyword/TF-IDF vectoriser written in plain numpy (embedding.py), and the
    default "answer" is built by quoting the retrieved passages (generate.py).
    So every notebook, lab, and test passes on a bare laptop. Flip one switch
    and it calls real neural embeddings and a real LLM instead.

  * Every step is a few lines you can actually read. The famous frameworks
    (LangChain, Chroma, FAISS) do exactly what this does -- once you have read
    this, their docs stop being magic. knowledge/10 shows the one-to-one map.
"""

from __future__ import annotations

__version__ = "0.1.0"

# The similarity below which we refuse to answer ("I don't know"). Tuned for the
# offline TF-IDF embedder on the sample docs; see generate.py and knowledge/08.
# It is deliberately modest: lexical similarity scores are small numbers, and we
# lean on a SECOND guardrail too (the answer refuses if no retrieved sentence
# shares a real word with the question). With neural embeddings you would raise
# this to ~0.3, because their scores for a good match are much higher.
DEFAULT_MIN_SCORE = 0.08

__all__ = ["__version__", "DEFAULT_MIN_SCORE"]
