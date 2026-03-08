# main.py
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, List, Dict, Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from models.embedder_loader import load_embedder, embed_query
from models.gmm_loader import load_gmm, get_query_cluster_probs
from models.chroma_loader import get_chroma_collection
from semantic_cache import SemanticCache


# ---------- Pydantic API models ----------

class QueryRequest(BaseModel):
    query: str

class QueryResultItem(BaseModel):
    id: str
    label_name: str
    snippet: str
    distance: float

class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str] = None
    similarity_score: Optional[float] = None
    result: Any
    dominant_cluster: Optional[int] = None

class CacheStatsResponse(BaseModel):
    total_queries: int
    hit_count: int
    miss_count: int
    hit_rate: float
    total_entries: int
    per_cluster_sizes: Optional[Dict[int, int]] = None


# ---------- Lifespan ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    embedder = load_embedder()
    gmm_model = load_gmm()
    chroma_collection = get_chroma_collection()

    semantic_cache = SemanticCache(
        gmm_model=gmm_model,
        embedder=embedder,
        similarity_threshold=0.85,
        allow_cross_cluster=False,
        max_cache_size=10_000,
    )

    app.state.embedder = embedder
    app.state.gmm_model = gmm_model
    app.state.chroma_collection = chroma_collection
    app.state.semantic_cache = semantic_cache

    yield

app = FastAPI(
    title="Trademarkia Semantic Cache Demo",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------- Retrieval helper ----------

def run_retrieval(query_embedding: np.ndarray, top_k: int = 5) -> List[QueryResultItem]:
    chroma_collection = app.state.chroma_collection

    results = chroma_collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["metadatas", "distances", "documents"],
    )

    items: List[QueryResultItem] = []
    ids = results.get("ids", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for i in range(len(ids)):
        doc_text = documents[i] or ""
        snippet = (doc_text[:200] + "...") if len(doc_text) > 200 else doc_text
        label_name = metadatas[i].get("label_name", "unknown") if metadatas and metadatas[i] else "unknown"

        items.append(
            QueryResultItem(
                id=str(ids[i]),
                label_name=label_name,
                snippet=snippet,
                distance=float(distances[i]),
            )
        )

    return items


# ---------- Optional root ----------

@app.get("/")
async def root():
    return {
        "message": "Trademarkia Semantic Cache API",
        "endpoints": ["/query", "/cache/stats", "/cache"],
    }


# ---------- Endpoints ----------

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(payload: QueryRequest) -> QueryResponse:
    query_text = payload.query

    embedder = app.state.embedder
    gmm_model = app.state.gmm_model
    semantic_cache: SemanticCache = app.state.semantic_cache

    query_embedding = embed_query(embedder, query_text)
    cluster_probs = get_query_cluster_probs(gmm_model, query_embedding)
    dominant_cluster = int(np.argmax(cluster_probs))

    cached_result, meta = semantic_cache.get(query_text)

    cache_hit: bool = meta.get("cache_hit", False)
    matched_query: Optional[str] = meta.get("matched_query")
    similarity_score: Optional[float] = meta.get("similarity_score")
    dominant_cluster_meta: Optional[int] = meta.get("dominant_cluster", dominant_cluster)

    result: Any = cached_result

    if not cache_hit:
        retrieval_results = run_retrieval(query_embedding, top_k=5)
        result = retrieval_results
        semantic_cache.add(query_text, result)

    return QueryResponse(
        query=query_text,
        cache_hit=cache_hit,
        matched_query=matched_query,
        similarity_score=similarity_score,
        result=result,
        dominant_cluster=dominant_cluster_meta,
    )

@app.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats_endpoint() -> CacheStatsResponse:
    semantic_cache: SemanticCache = app.state.semantic_cache
    stats = semantic_cache.get_stats()

    return CacheStatsResponse(
        total_queries=stats["total_queries"],
        hit_count=stats["hit_count"],
        miss_count=stats["miss_count"],
        hit_rate=stats["hit_rate"],
        total_entries=stats["total_entries"],
        per_cluster_sizes=stats.get("per_cluster_sizes"),
    )

@app.delete("/cache")
async def cache_clear_endpoint():
    semantic_cache: SemanticCache = app.state.semantic_cache
    semantic_cache.clear()
    return {
        "status": "cache cleared",
        "timestamp": datetime.now().isoformat(),
    }
