# semantic_cache.py
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.mixture import GaussianMixture
from sentence_transformers import SentenceTransformer
from collections import defaultdict, deque
import uuid
import time


class CacheEntry:
    """Simple container for cached query information."""

    def __init__(
        self,
        query_text: str,
        embedding: np.ndarray,
        cluster_probs: np.ndarray,
        result: Any,  # Any so we can store lists/dicts for FastAPI
        timestamp: float,
    ):
        self.id = str(uuid.uuid4())
        self.query_text = query_text
        self.embedding = embedding
        self.cluster_probs = cluster_probs
        self.result = result
        self.timestamp = timestamp  # for possible LRU logic later


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class SemanticCache:
    """
    In‑memory semantic cache using embeddings + GMM clusters.

    Intended interface for FastAPI later:
      - get(query_text) -> (result or None, metadata dict)
      - add(query_text, result)
      - get_stats()
      - clear()
    """

    def __init__(
        self,
        gmm_model: GaussianMixture,
        embedder: SentenceTransformer,
        similarity_threshold: float = 0.85,
        allow_cross_cluster: bool = False,
        max_cache_size: int = 10_000,
    ):
        self.gmm_model = gmm_model
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.allow_cross_cluster = allow_cross_cluster
        self.max_cache_size = max_cache_size

        # cluster_id -> list[CacheEntry]
        self._cluster_cache: Dict[int, List[CacheEntry]] = defaultdict(list)
        # global list to support simple FIFO eviction
        self._global_cache: deque[CacheEntry] = deque()

        # stats
        self.hit_count = 0
        self.miss_count = 0
        self.total_queries = 0

    # ---------- internal helpers ----------

    def _embed_query(self, query_text: str) -> np.ndarray:
        """Embed a single query string."""
        emb = self.embedder.encode([query_text])
        return np.asarray(emb[0], dtype=np.float32)

    def _get_cluster_probs(self, embedding: np.ndarray) -> np.ndarray:
        """Get GMM cluster probabilities for a query embedding."""
        probs = self.gmm_model.predict_proba(embedding.reshape(1, -1))[0]
        return np.asarray(probs, dtype=np.float32)

    @staticmethod
    def _dominant_cluster(cluster_probs: np.ndarray) -> int:
        return int(np.argmax(cluster_probs))

    def _evict_if_necessary(self) -> None:
        """Simple FIFO eviction across all clusters when size exceeds max_cache_size."""
        while len(self._global_cache) > self.max_cache_size:
            oldest = self._global_cache.popleft()
            # remove from its cluster list as well
            dom_cluster = self._dominant_cluster(oldest.cluster_probs)
            cluster_list = self._cluster_cache.get(dom_cluster, [])
            self._cluster_cache[dom_cluster] = [
                e for e in cluster_list if e.id != oldest.id
            ]
            if len(self._cluster_cache[dom_cluster]) == 0:
                del self._cluster_cache[dom_cluster]

    class _Match:
        """Tiny helper object to return best match info."""

        def __init__(self, entry: CacheEntry, similarity: float, cluster_idx: int):
            self.entry = entry
            self.similarity = similarity
            self.cluster_idx = cluster_idx

    def _find_best_match(
        self,
        query_embedding: np.ndarray,
        candidate_clusters: List[int],
    ) -> Optional["_Match"]:
        """Search only within candidate_clusters and return best similarity."""
        best_sim = -1.0
        best_entry: Optional[CacheEntry] = None
        best_cluster = -1

        for c_idx in candidate_clusters:
            entries = self._cluster_cache.get(int(c_idx), [])
            if not entries:
                continue
            for entry in entries:
                sim = cosine_similarity(query_embedding, entry.embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry
                    best_cluster = int(c_idx)

        if best_entry is None:
            return None
        return SemanticCache._Match(best_entry, best_sim, best_cluster)

    # ---------- public API ----------

    def get(self, query_text: str) -> Tuple[Optional[Any], Dict[str, Any]]:
        """
        Lookup in cache.

        Returns:
          result: cached result or None
          metadata: always includes
            - 'hit': bool
            - 'cache_hit': bool  (alias, for FastAPI/debug)
            - 'matched_query': str or None
            - 'similarity_score': float or None
            - 'dominant_cluster': int
            - 'reason': str
        """
        self.total_queries += 1

        embedding = self._embed_query(query_text)
        cluster_probs = self._get_cluster_probs(embedding)
        dom_cluster = self._dominant_cluster(cluster_probs)

        # 1) dominant‑cluster search
        match = self._find_best_match(embedding, [dom_cluster])
        if match and match.similarity >= self.similarity_threshold:
            self.hit_count += 1
            return match.entry.result, {
                "hit": True,
                "cache_hit": True,
                "matched_query": match.entry.query_text,
                "similarity_score": float(match.similarity),
                "dominant_cluster": dom_cluster,
                "reason": f"match in dominant cluster {dom_cluster}",
            }

        # 2) optional cross‑cluster fallback (top‑3 clusters)
        if self.allow_cross_cluster:
            top3 = list(np.argsort(cluster_probs)[-3:])
            # make sure dominant cluster is included only once
            if dom_cluster not in top3:
                top3.append(dom_cluster)
            match = self._find_best_match(embedding, top3)
            if match and match.similarity >= self.similarity_threshold:
                self.hit_count += 1
                return match.entry.result, {
                    "hit": True,
                    "cache_hit": True,
                    "matched_query": match.entry.query_text,
                    "similarity_score": float(match.similarity),
                    "dominant_cluster": dom_cluster,
                    "reason": "cross‑cluster match (top‑3 clusters)",
                }

        # miss
        self.miss_count += 1
        return None, {
            "hit": False,
            "cache_hit": False,
            "matched_query": None,
            "similarity_score": float(match.similarity) if match else None,
            "dominant_cluster": dom_cluster,
            "reason": "no semantic match above threshold",
        }

    def add(self, query_text: str, result: Any) -> None:
        """
        Add a new query+result to cache.

        Computes embedding + cluster_probs internally so caller can stay simple.
        """
        embedding = self._embed_query(query_text)
        cluster_probs = self._get_cluster_probs(embedding)
        dom_cluster = self._dominant_cluster(cluster_probs)

        entry = CacheEntry(
            query_text=query_text,
            embedding=embedding,
            cluster_probs=cluster_probs,
            result=result,
            timestamp=time.time(),
        )

        self._cluster_cache[dom_cluster].append(entry)
        self._global_cache.append(entry)
        self._evict_if_necessary()

    def get_stats(self) -> Dict[str, Any]:
        """Return stats for future /cache/stats endpoint."""
        return {
            "total_queries": self.total_queries,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": (
                self.hit_count / self.total_queries if self.total_queries > 0 else 0.0
            ),
            "total_entries": len(self._global_cache),
            "clusters_used": len(self._cluster_cache),
            "per_cluster_sizes": {int(k): len(v) for k, v in self._cluster_cache.items()},
            "params": {
                "similarity_threshold": self.similarity_threshold,
                "allow_cross_cluster": self.allow_cross_cluster,
                "max_cache_size": self.max_cache_size,
            },
        }

    def clear(self) -> None:
        """Wipe cache and stats."""
        self._cluster_cache.clear()
        self._global_cache.clear()
        self.hit_count = 0
        self.miss_count = 0
        self.total_queries = 0
