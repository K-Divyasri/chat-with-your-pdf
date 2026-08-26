"""Step 5 of RAG: given a question, fetch the passages most likely to answer it.

This is a thin but important layer. It ties the embedder to the store: embed the
question with the SAME embedder that embedded the chunks (this is essential --
the query and the documents must live in the same vector space or the geometry is
meaningless), then ask the store for the top-k closest chunks.

The result carries the similarity scores forward, because the next step
(generate.py) uses the best score to decide whether it is confident enough to
answer at all.
"""

from __future__ import annotations

from dataclasses import dataclass

from .chunking import Chunk
from .embedding import Embedder
from .vectorstore import VectorStore


@dataclass
class Retrieved:
    """A chunk that came back from a search, plus how well it matched (0..1)."""

    chunk: Chunk
    score: float


class Retriever:
    """Embeds a query and pulls the closest chunks out of a VectorStore."""

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def retrieve(self, question: str, k: int = 4) -> list[Retrieved]:
        """Return the top-`k` chunks for `question`, most similar first."""
        query_vector = self.embedder.embed(question)
        hits = self.store.search(query_vector, k=k)
        return [Retrieved(chunk=chunk, score=score) for chunk, score in hits]
