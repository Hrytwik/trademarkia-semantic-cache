import chromadb
from chromadb.config import Settings


def main():
    # Connect to the same persistent Chroma DB that was populated in Part 1.
    # Using PersistentClient means the index survives process restarts and
    # can be reused by later phases (GMM clustering, semantic cache, FastAPI).
    client = chromadb.PersistentClient(
        path="chroma_20newsgroups",
        settings=Settings(anonymized_telemetry=False),  # disable telemetry for reproducibility
    )

    # Retrieve the existing collection instead of recreating it.
    # This collection already contains:
    # - ids: "train_<index>"
    # - embeddings: 384‑dim sentence vectors
    # - metadata: {label_idx, label_name, snippet}
    collection = client.get_collection("twenty_newsgroups")

    # This script is a thin debug / smoke test:
    # given a natural language query, make sure the nearest neighbors
    # are semantically aligned (labels + snippets match the intent).
    query = "space shuttle launch and NASA mission"
    results = collection.query(
        query_texts=[query],
        n_results=5,
        # Note: we rely on Chroma's default embedding function here purely
        # for debugging. In the main pipeline, embeddings are generated
        # explicitly with sentence-transformers and written to the index.
    )

    print("Query:", query)
    for i in range(len(results["ids"][0])):
        doc_id = results["ids"][0][i]
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        snippet = meta.get("snippet", "")[:120]

        # For each hit, we surface:
        # - the document id (ties back to artifacts on disk),
        # - the true newsgroup label (sanity‑check the retrieval),
        # - the distance in embedding space (lower = closer),
        # - a short text snippet for quick qualitative inspection.
        print(f"\n{i+1}. id={doc_id}, label={meta['label_name']}, distance={dist:.4f}")
        print("   snippet:", snippet)


if __name__ == "__main__":
    main()
