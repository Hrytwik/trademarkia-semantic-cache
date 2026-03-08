# services/cache_service.py
from typing import Tuple, Dict, Any, Optional
from semantic_cache import SemanticCache

def handle_cache_lookup(
    cache: SemanticCache,
    query_text: str
) -> Tuple[bool, Optional[str], Optional[float], Any, Optional[int]]:
    """
    Wrapper for SemanticCache.get() that extracts metadata for API response.
    
    Returns: (cache_hit, matched_query, similarity_score, result, dominant_cluster)
    """
    result, metadata = cache.get(query_text)
    
    if metadata is None:
        return False, None, None, None, None
    
    cache_hit = metadata.get("cache_hit", False)
    matched_query = metadata.get("matched_query")
    similarity_score = metadata.get("similarity_score")
    dominant_cluster = metadata.get("dominant_cluster")
    result_data = result if result else None
    
    return cache_hit, matched_query, similarity_score, result_data, dominant_cluster
