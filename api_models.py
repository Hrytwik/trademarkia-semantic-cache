# api_models.py
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

class QueryResultItem(BaseModel):
    id: str
    label_name: str
    snippet: str
    distance: float  # cosine distance from Chroma (lower = better)

class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str] = None
    similarity_score: Optional[float] = None
    result: Any  # str for cache hit, List[QueryResultItem] for miss
    dominant_cluster: Optional[int] = None

class CacheStatsResponse(BaseModel):
    total_queries: int
    hit_count: int
    miss_count: int
    hit_rate: float
    total_entries: int
    # Plus whatever else your get_stats() returns
    per_cluster_sizes: Optional[Dict[int, int]] = None
