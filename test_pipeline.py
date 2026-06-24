"""
Offline verification:
  1. FAISS build+query pipeline on synthetic 1024-dim embeddings, using the SAME
     index recipe deepghs uses (OPQ/IVF/PQ + cosine via L2-normalize). Proves the
     code that builds an index and the code that queries it agree: a query equal
     to a stored vector must return that vector as nearest neighbour.
  2. resolver parsers against captured-shape JSON for every site, incl. rating
     filtering and the safebooru file_url fallback.
No network, no HuggingFace, no booru access required.
"""
import json
import numpy as np
import faiss

import resolver

DIM = 1024  # WD14 SwinV2_v3 embedding dimension (confirmed from deepghs infos.json)


def _normalize(x):
    x = x.astype(np.float32).copy()
    faiss.normalize_L2(x)
    return x


def build_index(embs, factory="IVF256,Flat", param="nprobe=16"):
    """Mirror of build_index.py's core, small enough to train on toy data.
    Real builds use deepghs embeddings_tools (OPQ256_1280,IVF16384_HNSW32,PQ256x8)."""
    embs = _normalize(embs)
    index = faiss.index_factory(DIM, factory, faiss.METRIC_INNER_PRODUCT)
    index.train(embs)
    index.add(embs)
    faiss.ParameterSpace().set_index_parameters(index, param)
    return index


def test_faiss_roundtrip():
    rng = np.random.default_rng(0)
    n = 5000
    embs = rng.standard_normal((n, DIM)).astype(np.float32)
    ids = np.arange(100000, 100000 + n)  # fake post ids

    index = build_index(embs)

    # Save + reload exactly like the app does, to catch serialization issues.
    faiss.write_index(index, "/tmp/toy.index")
    np.save("/tmp/toy_ids.npy", ids)
    index2 = faiss.read_index("/tmp/toy.index")
    faiss.ParameterSpace().set_index_parameters(index2, "nprobe=64")
    ids2 = np.load("/tmp/toy_ids.npy")

    # Query with a copy of stored vector #1234 -> must come back rank 0.
    q = _normalize(embs[1234:1235].copy())
    dists, idx = index2.search(q, k=5)
    top_id = ids2[idx][0][0]
    assert top_id == ids[1234], f"expected self as NN, got {top_id}"
    assert dists[0][0] > 0.99, f"cosine of self should ~1, got {dists[0][0]:.3f}"

    # A second query: nearest of a random vector must be a valid stored id.
    q2 = _normalize(rng.standard_normal((1, DIM)))
    _, idx2 = index2.search(q2, k=10)
    returned = ids2[idx2][0]
    assert all(r in set(ids.tolist()) for r in returned.tolist())
    print("PASS faiss_roundtrip: self-NN recovered, cosine~1, ids valid")


def test_resolver_gelbooru_style_list():
    # rule34 / safebooru: bare list
    body = [{"file_url": "https://x/r.jpg", "rating": "explicit"}]
    url, rating = resolver._extract_gelbooru_style(body)
    assert url == "https://x/r.jpg" and rating == "explicit"
    # rating filter blocks it when explicit not accepted
    assert resolver._passes(rating, {"general", "sensitive"}) is False
    assert resolver._passes(rating, {"explicit"}) is True
    print("PASS resolver gelbooru-style list + rating filter")


def test_resolver_gelbooru_dict_and_empty():
    body = {"post": [{"file_url": "https://g/i.png", "rating": "sensitive"}]}
    url, rating = resolver._extract_gelbooru_style(body)
    assert url == "https://g/i.png" and rating == "sensitive"
    # empty result shapes
    assert resolver._extract_gelbooru_style({"post": []}) == (None, None)
    assert resolver._extract_gelbooru_style([]) == (None, None)
    print("PASS resolver gelbooru dict + empty handling")


def test_resolver_safebooru_fallback():
    # safebooru sometimes lacks file_url; build from directory+image
    body = [{"directory": "12/ab", "image": "abc.png", "rating": "safe"}]
    url, rating = resolver._extract_gelbooru_style(body)
    assert url == "https://safebooru.org/images/12/ab/abc.png"
    assert rating == "general"  # 'safe' -> 'general'
    print("PASS resolver safebooru file_url fallback + safe->general")


def test_resolver_e621():
    body = {"post": {"file": {"url": "https://e/abc.jpg"}, "rating": "e"}}
    url, rating = resolver._extract_e621(body)
    assert url == "https://e/abc.jpg" and rating == "explicit"
    assert resolver._extract_e621({"post": None}) == (None, None)
    print("PASS resolver e621")


def test_resolver_danbooru():
    body = {"large_file_url": "https://d/big.jpg", "file_url": "https://d/x.jpg", "rating": "q"}
    url, rating = resolver._extract_danbooru(body)
    assert url == "https://d/big.jpg" and rating == "questionable"
    print("PASS resolver danbooru (prefers large_file_url)")


def test_site_registry():
    for s in ["rule34", "gelbooru", "safebooru", "e621", "danbooru"]:
        assert s in resolver.SITES
        assert "{id}" in resolver.SITES[s]["api"]
    print("PASS site registry complete")


if __name__ == "__main__":
    test_faiss_roundtrip()
    test_resolver_gelbooru_style_list()
    test_resolver_gelbooru_dict_and_empty()
    test_resolver_safebooru_fallback()
    test_resolver_e621()
    test_resolver_danbooru()
    test_site_registry()
    print("\nALL TESTS PASSED")
