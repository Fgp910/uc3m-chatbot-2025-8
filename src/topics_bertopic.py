from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import hdbscan
import os
from collections import Counter

from src.vector_store import get_vectorstore
import logging
logger = logging.getLogger(__name__)

def _load_corpus_from_chroma(max_docs: int | None = None) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Extracts raw chunk texts + metadatas from the persisted Chroma collection.
    """
    vs = get_vectorstore()
    col = vs._collection

    data = col.get(
        include=["documents", "metadatas"],
        limit=max_docs
    )

    docs = data.get("documents", []) or []
    metas = data.get("metadatas", []) or []
    # Filter empty docs
    filtered = [(d, m) for d, m in zip(docs, metas) if d and isinstance(d, str) and d.strip()]
    docs = [d for d, _ in filtered]
    metas = [m for _, m in filtered]
    return docs, metas


def train_bertopic_from_chroma(
    model_dir: str = "output/bertopic",
    embedding_model_name: str = "all-MiniLM-L6-v2",
    max_docs: int | None = None,
    min_topic_size: int = 30,
) -> None:
    """
    Trains BERTopic on Chroma chunk texts and saves it to disk.
    """
    texts, _ = _load_corpus_from_chroma(max_docs=max_docs)
    if len(texts) < 50:
        raise ValueError(f"Not enough texts to train topics. Got {len(texts)}.")

    embedding_model = SentenceTransformer(embedding_model_name)

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=min_topic_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(texts)
    os.makedirs(os.path.dirname(model_dir), exist_ok=True)
    topic_model.save(model_dir)


def load_bertopic(model_dir: str = "output/bertopic_model.pkl") -> BERTopic:
    return BERTopic.load(model_dir)


def top_topics_for_query(
    topic_model: BERTopic,
    query: str,
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    """
    Returns top-N topics for a query:
    [{"topic_id": int, "prob": float, "keywords": [("word", weight), ...]}]
    """
    topics, probs = topic_model.transform([query])
    t = int(topics[0])
    p = float(probs[0][t]) if probs is not None and t >= 0 and t < probs.shape[1] else 0.0

    # If model assigns -1 (outlier), we still can provide nearest topics by similarity:
    if t == -1:
        return nearest_topics_by_embedding(topic_model, query, top_n=top_n)

    out = [{
        "topic_id": t,
        "prob": p,
        "keywords": topic_model.get_topic(t)[:10] if topic_model.get_topic(t) else [],
    }]

    # Add additional nearest topics as “related”
    related = nearest_topics_by_embedding(topic_model, query, top_n=top_n)
    # merge unique topic_ids
    seen = {t}
    for r in related:
        if r["topic_id"] not in seen:
            out.append(r)
            seen.add(r["topic_id"])
        if len(out) >= top_n:
            break

    return out[:top_n]


def nearest_topics_by_embedding(topic_model: BERTopic, query: str, top_n: int = 3):
    """
    Compute cosine similarity between query embedding and topic embeddings
    and return top-N topic ids with keywords.
    """
    # topic embeddings exist after training
    topic_embs = topic_model.topic_embeddings_
    if topic_embs is None:
        return []

    # Get query embedding from the embedding backend (works for SentenceTransformerBackend)
    emb_model = topic_model.embedding_model
    if hasattr(emb_model, "embed_queries"):
        q_emb = emb_model.embed_queries([query])[0]
    elif hasattr(emb_model, "embed_documents"):
        q_emb = emb_model.embed_documents([query])[0]
    elif hasattr(emb_model, "encode"):
        q_emb = emb_model.encode([query], normalize_embeddings=True)[0]
    else:
        raise AttributeError(f"Unsupported embedding model backend: {type(emb_model)}")

    # Normalize query embedding
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)

    # Normalize topic embeddings
    te = topic_embs / (np.linalg.norm(topic_embs, axis=1, keepdims=True) + 1e-12)
    sims = te @ q_emb

    # Topic 0..N-1 might include -1 mapping; get valid topic ids from topic info
    info = topic_model.get_topic_info()
    valid_topic_ids = [int(x) for x in info["Topic"].tolist() if int(x) >= 0]

    # Rank valid topics
    ranked = sorted(valid_topic_ids, key=lambda tid: float(sims[tid]), reverse=True)[:top_n]
    out = []
    for tid in ranked:
        out.append({
            "topic_id": tid,
            "prob": float(sims[tid]),
            "keywords": topic_model.get_topic(tid)[:10] if topic_model.get_topic(tid) else [],
        })
    return out


def suggest_questions_from_topics(topics: List[Dict[str, Any]], n: int = 5) -> List[str]:
    """
    Simple template-based suggestions (no LLM needed).
    """
    suggestions = []
    for t in topics:
        kws = [w for (w, _) in t.get("keywords", [])[:5]]
        if not kws:
            continue
        kw1 = kws[0]
        kw2 = kws[1] if len(kws) > 1 else kws[0]
        suggestions.extend([
            f"What does '{kw1}' mean in this context?",
            f"Can you summarize the main points about '{kw1}'?",
            f"What's the difference between '{kw1}' and '{kw2}'?",
            f"Where in the documents is '{kw1}' discussed?",
        ])
    # unique + cut
    seen = set()
    out = []
    for s in suggestions:
        if s not in seen:
            out.append(s)
            seen.add(s)
        if len(out) >= n:
            break
    return out


def topics_from_retrieved_chunks(topic_model, retriever, query: str, top_n: int = 3):
    # 1) retrieve chunks (same retriever as the RAG)
    docs = retriever.invoke(query)
    texts = [d.page_content for d in docs if getattr(d, "page_content", None)]

    if not texts:
        return []

    # 2) infer topic for each retrieved chunk
    ts, _ = topic_model.transform(texts)

    # 3) pick most frequent topics (ignore outlier -1)
    cnt = Counter([int(t) for t in ts if int(t) != -1])
    if not cnt:
        return []

    top_topic_ids = [tid for tid, _ in cnt.most_common(top_n)]

    # 4) build display objects with keywords
    out = []
    for tid in top_topic_ids:
        kws = topic_model.get_topic(tid) or []
        out.append({
            "topic_id": tid,
            "prob": cnt[tid] / max(1, len(texts)),  # frequency proxy
            "keywords": kws[:10],
        })
    return out


def merge_intent_and_grounded_topics(
    query_topics: List[Dict[str, Any]],
    chunks_topics: List[Dict[str, Any]],
    top_n: int = 5,
    w_chunks: float = 0.65,
    w_query: float = 0.35,
) -> List[Dict[str, Any]]:
    """
    Merge query-based topics (intent) with chunk-based topics (grounded evidence).

    query_topics: output of top_topics_for_query(...)
    chunks_topics: output of topics_from_retrieved_chunks(...)

    Returns:
        [
          {
            "topic_id": int,
            "score": float,
            "query_score": float,
            "chunks_score": float,
            "keywords": [...]
          }
        ]
    """

    logger.info("🔀 Merging query topics and chunk topics")
    merged: Dict[int, Dict[str, Any]] = {}

    # ----------------------------
    # 1) Add grounded (chunks) topics first
    # ----------------------------
    for t in chunks_topics:
        tid = int(t.get("topic_id", -1))
        if tid == -1:
            logger.debug("Skipping outlier topic -1 from chunks")
            continue

        merged[tid] = {
            "topic_id": tid,
            "chunks_score": float(t.get("prob", 0.0)),
            "query_score": 0.0,
            "keywords": t.get("keywords", []),
        }

    # ----------------------------
    # 2) Add query (intent) topics
    # ----------------------------
    for t in query_topics:
        tid = int(t.get("topic_id", -1))
        if tid == -1:
            logger.debug("Skipping outlier topic -1 from query")
            continue

        if tid not in merged:
            merged[tid] = {
                "topic_id": tid,
                "chunks_score": 0.0,
                "query_score": 0.0,
                "keywords": t.get("keywords", []),
            }

        merged[tid]["query_score"] = float(t.get("prob", 0.0))

        # Prefer keywords if chunks didn't provide them
        if not merged[tid]["keywords"] and t.get("keywords"):
            merged[tid]["keywords"] = t["keywords"]

    if not merged:
        logger.warning("⚠️ No topics to merge (both lists empty?)")
        return []

    # ----------------------------
    # 3) Normalize query scores
    # ----------------------------
    query_vals = [v["query_score"] for v in merged.values()]
    q_min, q_max = min(query_vals), max(query_vals)

    for v in merged.values():
        if q_max - q_min < 1e-9:
            v["query_norm"] = 0.0
        else:
            v["query_norm"] = (v["query_score"] - q_min) / (q_max - q_min)

    # ----------------------------
    # 4) Final combined score
    # ----------------------------
    for v in merged.values():
        v["score"] = (
            w_chunks * v["chunks_score"] +
            w_query * v["query_norm"]
        )

    # ----------------------------
    # 5) Rank & return
    # ----------------------------
    ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)

    return dedupe_topics_by_main_keyword(ranked[:top_n])


import re

STOP_KEYWORDS = {
    "var", "vars", "variable", "etc", "e.g", "ie", "i.e", "llc", "com"
}

def dedupe_topics_by_main_keyword(topics):
    used = set()
    out = []
    for t in topics:
        kws = clean_topic_keywords(t.get("keywords", []), max_kw=6)
        if not kws:
            continue
        main = kws[0][0].strip().lower()  # keyword principal
        if main in used:
            continue
        used.add(main)
        t["keywords"] = kws
        out.append(t)
    return out

def clean_topic_keywords(
    keywords,
    max_kw: int = 6,
    drop_numbers: bool = True,
    drop_short: bool = True,
) :
    """
    keywords: list[tuple[str, float]] de BERTopic
    returns: list[tuple[str, float]] limpio
    """
    seen = set()
    out = []

    for w, score in keywords:
        if not w:
            continue

        w2 = w.strip().lower()

        # drop pure numbers or number-like tokens
        if drop_numbers and (w2.isdigit() or re.fullmatch(r"\d+(\.\d+)?", w2)):
            continue

        # drop tokens containing mostly digits (e.g., 713, 207, 713 207)
        if drop_numbers and sum(ch.isdigit() for ch in w2) >= max(2, len(w2)//2):
            continue

        # drop very short junk
        if drop_short and len(w2) <= 2:
            continue

        # drop stop keywords (custom)
        if w2 in STOP_KEYWORDS:
            continue

        # dedupe within topic
        if w2 in seen:
            continue

        seen.add(w2)
        out.append((w, score))

        if len(out) >= max_kw:
            break

    return out
