# Trademarkia Semantic Cache – 20 Newsgroups

Lightweight semantic search and **semantic cache** built on top of the 20 Newsgroups dataset using sentence embeddings, Gaussian Mixture clustering, a custom cache, and a FastAPI service.

This project is my solution for the **AI Engineer** assignment at Trademarkia.

---

## 1. Project Overview

The system turns the classic 20 Newsgroups corpus (~11k documents, 20 topics) into a semantic search service with an intelligent cache:

1. **Embeddings (Phase 1)**  
   - Load 20 Newsgroups train split via `sklearn.datasets.fetch_20newsgroups(subset="train", remove=("headers","footers","quotes"))`.  
   - Clean raw posts (lowercase, strip control characters / email noise, normalize whitespace, keep meaningful punctuation).  
   - Encode each document using **SentenceTransformer**:  
     - Model: `all-MiniLM-L6-v2` (384‑dim sentence embeddings).  
   - Save artifacts under `artifacts/`:
     - `embeddings.npy`, `ids.npy`, `label_indices.npy`, `label_names.npy`, `texts_clean.npy`.  
   - Build a persistent **Chroma** vector index `twenty_newsgroups` under `./chroma_20newsgroups` using cosine distance.

2. **Fuzzy Clustering (Phase 2)**  
   - Run a **Gaussian Mixture Model (GMM)** on the embeddings:
     - `n_components = 20`, `covariance_type = "diag"`, `max_iter = 300`, `random_state = 42`.  
   - Output:
     - `gmm_diag20_model.pkl` – fitted GMM (saved via `joblib`).  
     - `gmm_diag20_cluster_proba.npy` – per‑document soft memberships (distribution over 20 clusters).  
     - `gmm_diag20_cluster_labels_hard.npy` – hard labels for analysis only.  
   - This gives **soft/fuzzy clusters**: each document belongs to multiple topics to different degrees instead of a single hard label.

3. **Semantic Cache (Phase 3)**  
   - Implemented a **home‑rolled semantic cache** (no Redis / Memcached) in `semantic_cache.py`.  
   - Key ideas:
     - Each incoming query is embedded with the same `all-MiniLM-L6-v2` model.  
     - The trained GMM produces a probability distribution over 20 clusters for the query.  
     - Cache entries are stored by **dominant cluster**, using cosine similarity in embedding space to decide reuse.  
   - The cache decides whether two differently phrased queries are “close enough” to share results using:
     - Cosine similarity threshold (e.g. 0.85, 0.78, 0.70).  
     - Optional cross‑cluster lookup over top‑3 clusters by probability.  
   - All logic uses only Python data structures (`dict`, `list`, `deque`).

4. **FastAPI Service (Phase 4)**  
   - Exposes the semantic cache + Chroma retrieval as a small REST API in `main.py`.  
   - Endpoints:
     - `POST /query` – embed + cluster + cache, falling back to Chroma retrieval on cache miss.  
     - `GET /cache/stats` – live cache statistics.  
     - `DELETE /cache` – flush cache and reset stats.  

---

## 2. Tech Stack

- **Language**: Python 3.11  
- **ML / NLP**:
  - `sentence-transformers` (SentenceTransformer `all-MiniLM-L6-v2`)
  - `scikit-learn` (GaussianMixture, metrics)
- **Vector DB**: `chromadb` (persistent client, cosine distance)  
- **API**: `FastAPI`, `uvicorn`  
- **Other**: `numpy`, `joblib`, `pydantic`

---

## 3. Setup

### 3.1 Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3.2 Install dependencies

```bash
pip install -r requirements.txt
```

If there is no `requirements.txt`, you can install directly:

```bash
pip install fastapi uvicorn chromadb sentence-transformers scikit-learn joblib numpy
```

> Note: Phases 1–3 should already have produced `artifacts/` and `./chroma_20newsgroups/`.  
> If you clone only this repo, run the Phase‑1 and Phase‑2 scripts first to regenerate artifacts and the Chroma index.

---

## 4. Running the FastAPI Service

From the project root:

```bash
uvicorn main:app --reload
```

- API root: http://127.0.0.1:8000/
- Interactive docs (Swagger): http://127.0.0.1:8000/docs

---

## 5. API Endpoints

### 5.1 `POST /query`

**Request body**

```json
{
  "query": "machine learning algorithms"
}
```

**Behavior**

1. Embed the query with `all-MiniLM-L6-v2`.  
2. Compute query cluster probabilities using the trained GMM.  
3. Ask the `SemanticCache`:

   - If a **semantic match** is found (similarity ≥ threshold):
     - Return cached result (no retrieval needed).
   - Otherwise:
     - Run a semantic search against Chroma (`twenty_newsgroups`) using the query embedding.
     - Take top‑k documents with ids, label names, snippets, distances.
     - Store this list as the cache result for the query.
     - Return it to the client.

**Response example (cache miss → retrieval)**

```json
{
  "query": "machine learning algorithms",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": [
    {
      "id": "train_1907",
      "label_name": "comp.graphics",
      "snippet": "....",
      "distance": 0.63
    }
  ],
  "dominant_cluster": 15
}
```

If you repeat the same or a very similar query, `cache_hit` becomes `true` and `matched_query` shows which earlier query it reused.

---

### 5.2 `GET /cache/stats`

Returns live cache statistics:

```json
{
  "total_queries": 5,
  "hit_count": 3,
  "miss_count": 2,
  "hit_rate": 0.6,
  "total_entries": 2,
  "per_cluster_sizes": {
    "15": 2
  }
}
```

This is driven directly by `SemanticCache.get_stats()`.

---

### 5.3 `DELETE /cache`

Flushes the cache and resets stats:

```json
{
  "status": "cache cleared",
  "timestamp": "2026-03-08T18:15:00.000000"
}
```

---

## 6. How the Semantic Cache Works

The `SemanticCache` class (in `semantic_cache.py`) implements a **cluster-aware, embedding-based** cache:

- Each query is represented by:
  - The original text.
  - A 384‑D embedding.
  - A 20‑dim cluster probability vector from the GMM.
  - A stored `result` (list of retrieved docs).
- Cache entries are grouped by **dominant cluster id** in `_cluster_cache: Dict[int, List[CacheEntry]]`.  
- Lookup uses cosine similarity:

  1. Embed the new query and compute its cluster probabilities.  
  2. Find the dominant cluster and search for the best match only within that cluster.  
  3. If `allow_cross_cluster=True`, optionally search top‑3 clusters by probability.  
  4. If the best similarity ≥ `similarity_threshold`, it is a **cache hit**, otherwise a **miss**.

- Eviction is simple **FIFO** using a global `deque` when `max_cache_size` is exceeded.

By tuning `similarity_threshold` and `allow_cross_cluster`, the cache can be made:

- **Conservative** – high precision, fewer hits (higher threshold, no cross‑cluster).  
- **Balanced** – medium threshold, cross‑cluster allowed.  
- **Aggressive** – lower threshold, more reuse but more risk of over‑matching.

---

## 7. How to Run the Demo

Suggested quick demo flow:

1. Start the API:

   ```bash
   uvicorn main:app --reload
   ```

2. In Swagger (`/docs`), call `POST /query` with:

   - `"NASA space shuttle mission details"`  
   - Then a paraphrase like `"space shuttle launch by NASA"`

   Show that the second call becomes a **cache hit** with a high `similarity_score` and the first query shown as `matched_query`.

3. Call `GET /cache/stats` to show `hit_count`, `miss_count`, and `hit_rate`.

4. Call `DELETE /cache` and confirm that stats reset.

---

## 8. Notes

- All caching is implemented **from first principles** (no Redis/Memcached).  
- Clustering is fully **unsupervised**; dataset labels are used only for offline analysis, not for cache logic.  
- The design is modular:
  - `semantic_cache.py` – cache logic.  
  - `models/` – loaders for embedder, GMM, and Chroma.  
  - `main.py` – FastAPI service wiring everything together.
