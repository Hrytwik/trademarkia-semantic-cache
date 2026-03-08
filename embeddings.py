import os
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
from data_utils import clean_text


def load_texts():
    dataset = fetch_20newsgroups(
        subset="train",
        remove=("headers", "footers", "quotes"),
    )
    texts_raw = dataset.data
    labels = dataset.target
    label_names = dataset.target_names
    texts_clean = [clean_text(t) for t in texts_raw]
    return texts_clean, labels, label_names


def main():
    texts, labels, label_names = load_texts()
    print(f"Total docs: {len(texts)}")

    # 'all-MiniLM-L6-v2' is a compact sentence embedding model optimized for
    # semantic similarity. It keeps embeddings small (384 dims) so the downstream
    # vector index and GMM clustering stay lightweight.
    # Alternatives considered: 'all-mpnet-base-v2' (higher quality, 768 dims, heavier).
    # For this assignment, latency and index size matter more than marginal gains.

    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    print(f"Loaded model: {model_name}")

    # Encode all documents in batches
    batch_size = 64
    all_embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    print("All embeddings shape:", all_embeddings.shape)

    # Create IDs and metadata arrays
    ids = [f"train_{i}" for i in range(len(texts))]
    label_indices = np.array(labels)
    label_names_arr = np.array([label_names[i] for i in labels])

    # Ensure output directory exists
    os.makedirs("artifacts", exist_ok=True)

    # Save to disk for debugging / reuse
    np.save("artifacts/embeddings.npy", all_embeddings)
    np.save("artifacts/ids.npy", np.array(ids))
    np.save("artifacts/label_indices.npy", label_indices)
    np.save("artifacts/label_names.npy", label_names_arr)
    np.save("artifacts/texts_clean.npy", np.array(texts, dtype=object))

    print("Saved embeddings and metadata to artifacts/ directory")


if __name__ == "__main__":
    main()
