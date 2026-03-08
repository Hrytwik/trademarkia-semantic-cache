# models/embedder_loader.py
from sentence_transformers import SentenceTransformer
import numpy as np

def load_embedder() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")

def embed_query(embedder: SentenceTransformer, query_text: str) -> np.ndarray:
    return embedder.encode(query_text)
