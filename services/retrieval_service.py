# services/retrieval_service.py
from typing import List
import numpy as np
from chromadb.api import Collection
from sentence_transformers import SentenceTransformer
from api_models import QueryResultItem

def run_retrieval(
    query_embedding: np.ndarray,
    chroma_collection: Collection,
    top_k: int = 5,
    cluster_probs: Optional[np.ndarray] = None
) -> List[QueryResultItem]:
    """
    Perform semantic retrieval against Chroma index.
    
    Optionally restrict to top clusters for efficiency (Phase 4 enhancement).
    """
    # Query all docs for simplicity (matches your Phase 1 setup)
    # Later: filter by cluster if you add cluster metadata to Chroma
    results = chroma_collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
        include=["metadatas", "distances", "documents"]
    )
    
    top_k_docs = []
    for i in range(top_k):
        if i < len(results["ids"][0]):
            top_k_docs.append(QueryResultItem(
                id=results["ids"][0][i],
                label_name=results["metadatas"][0][i].get("label_name", "unknown"),
                snippet=results["documents"][0][i][:200] + "..." if results["documents"][0][i] else "",
                distance=results["distances"][0][i]
            ))
    
    return top_k_docs
