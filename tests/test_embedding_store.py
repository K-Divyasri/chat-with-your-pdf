"""Tests for the embedder and the vector store -- the geometry at the core of RAG."""

from __future__ import annotations

import numpy as np
import pytest

from pdf_chat.chunking import Chunk
from pdf_chat.embedding import HashingEmbedder, tokenize
from pdf_chat.vectorstore import VectorStore


def test_tokenize_drops_stopwords_and_lowercases():
    toks = tokenize("The Jupiter is THE largest Planet")
    assert "the" not in toks and "is" not in toks
    assert "jupiter" in toks and "largest" in toks and "planet" in toks


def test_embeddings_are_unit_length():
    emb = HashingEmbedder().fit(["jupiter is the largest planet"])
    v = emb.embed("jupiter is the largest planet")
    assert np.isclose(np.linalg.norm(v), 1.0)


def test_embedding_is_deterministic():
    # crc32 hashing (not Python's salted hash) => identical vectors across calls.
    emb = HashingEmbedder().fit(["some corpus text about coffee and planets"])
    a = emb.embed("coffee brewing guide")
    b = emb.embed("coffee brewing guide")
    assert np.array_equal(a, b)


def test_similar_text_scores_higher_than_unrelated():
    corpus = [
        "Jupiter is the largest planet with many moons",
        "coffee is brewed with hot water and ground beans",
    ]
    emb = HashingEmbedder().fit(corpus)
    q = emb.embed("how many moons does jupiter have")
    planet_sim = float(emb.embed(corpus[0]) @ q)
    coffee_sim = float(emb.embed(corpus[1]) @ q)
    assert planet_sim > coffee_sim


def test_empty_embed_many_has_right_shape():
    emb = HashingEmbedder(dim=64)
    out = emb.embed_many([])
    assert out.shape == (0, 64)


def _chunks(texts):
    return [Chunk(text=t, source="d.pdf", page_number=1, chunk_index=i)
            for i, t in enumerate(texts)]


def test_store_search_ranks_by_similarity():
    texts = ["vacation days and paid leave", "jupiter has many moons", "coffee ratio guide"]
    emb = HashingEmbedder().fit(texts)
    store = VectorStore(dim=emb.dim)
    store.add(_chunks(texts), emb.embed_many(texts))
    hits = store.search(emb.embed("how much vacation do i get"), k=3)
    assert len(hits) == 3
    assert hits[0][0].text == "vacation days and paid leave"
    # scores are sorted descending
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_store_add_rejects_mismatched_lengths():
    emb = HashingEmbedder(dim=32)
    store = VectorStore(dim=32)
    with pytest.raises(ValueError):
        store.add(_chunks(["a", "b"]), emb.embed_many(["a"]))


def test_store_save_and_load_round_trip(tmp_path):
    texts = ["vacation days", "jupiter moons", "coffee ratio"]
    emb = HashingEmbedder().fit(texts)
    store = VectorStore(dim=emb.dim)
    store.add(_chunks(texts), emb.embed_many(texts))
    store.save(tmp_path / "idx")
    reloaded = VectorStore.load(tmp_path / "idx")
    assert len(reloaded) == 3
    q = emb.embed("vacation")
    assert store.search(q, 1)[0][0].text == reloaded.search(q, 1)[0][0].text
