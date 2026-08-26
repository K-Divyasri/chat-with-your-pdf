"""The whole RAG pipeline in one object: load -> chunk -> embed -> store, then ask.

Everything in the other files is a single clean step. This file is the conductor
that runs them in order and hands you two verbs:

    rag = Rag()                      # choose the embedder and the settings
    rag.ingest(["data/handbook.pdf"])  # the slow part: read, chunk, embed, index
    answer = rag.ask("how many vacation days?")   # the fast part: retrieve + answer

Ingesting is done once; asking is done as many times as you like. You can also
`save()` the index to disk and `load()` it later so you do not re-embed every run.
"""

from __future__ import annotations

from pathlib import Path

from .chunking import Chunk, chunk_pages
from .embedding import Embedder, HashingEmbedder, make_embedder
from .generate import Answer, generate_answer
from .loading import load_any
from .retrieval import Retriever
from .vectorstore import VectorStore
from . import DEFAULT_MIN_SCORE


class Rag:
    """A ready-to-use retrieval-augmented-generation engine over your documents."""

    def __init__(
        self,
        *,
        embedder: Embedder | str = "hashing",
        chunk_size: int = 500,
        overlap: int = 100,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> None:
        self.embedder: Embedder = (
            make_embedder(embedder) if isinstance(embedder, str) else embedder
        )
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_score = min_score
        self.store = VectorStore(dim=self.embedder.dim)
        self.retriever = Retriever(self.embedder, self.store)

    # --- indexing -------------------------------------------------------------
    def ingest(self, paths: list[str | Path]) -> int:
        """Read, chunk, embed and index the given .pdf/.txt files. Returns #chunks added.

        For the hashing embedder we `fit` its IDF weights on the freshly-chunked
        text first, so common words in *these* documents get down-weighted.
        """
        pages = []
        for path in paths:
            pages.extend(load_any(path))
        chunks: list[Chunk] = chunk_pages(
            pages, chunk_size=self.chunk_size, overlap=self.overlap
        )
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        # Only the hashing embedder learns from the corpus; the neural one ignores fit.
        if isinstance(self.embedder, HashingEmbedder):
            self.embedder.fit(texts)
        vectors = self.embedder.embed_many(texts)
        self.store.add(chunks, vectors)
        return len(chunks)

    # --- querying -------------------------------------------------------------
    def ask(
        self,
        question: str,
        *,
        k: int = 4,
        offline: bool = True,
        model: str | None = None,
    ) -> Answer:
        """Answer `question` over the ingested documents, with citations or a refusal."""
        retrieved = self.retriever.retrieve(question, k=k)
        return generate_answer(
            question,
            retrieved,
            offline=offline,
            min_score=self.min_score,
            model=model,
        )

    # --- persistence ----------------------------------------------------------
    def save(self, folder: str | Path) -> None:
        """Save the vector index so you can reload without re-embedding."""
        self.store.save(folder)

    @classmethod
    def load(cls, folder: str | Path, *, embedder: Embedder | str = "hashing", **kwargs) -> "Rag":
        """Load a saved index. Pass the SAME kind of embedder you built it with."""
        rag = cls(embedder=embedder, **kwargs)
        rag.store = VectorStore.load(folder)
        # Re-fit the hashing embedder's IDF on the loaded chunks so queries match.
        if isinstance(rag.embedder, HashingEmbedder):
            rag.embedder.fit([c.text for c in rag.store._chunks])
        rag.retriever = Retriever(rag.embedder, rag.store)
        return rag

    def __len__(self) -> int:
        return len(self.store)
