import os
import numpy as np
import chromadb
from chromadb.config import Settings


def load_artifacts():
    embeddings = np.load("artifacts/embeddings.npy")
    ids = np.load("artifacts/ids.npy")
    label_indices = np.load("artifacts/label_indices.npy")
    label_names = np.load("artifacts/label_names.npy")
    texts_clean = np.load("artifacts/texts_clean.npy", allow_pickle=True)
    return embeddings, ids, label_indices, label_names, texts_clean


def main():
    embeddings, ids, label_indices, label_names, texts_clean = load_artifacts()
    print("Embeddings shape:", embeddings.shape)
    print("Number of ids:", len(ids))

    # Ensure persistence directory exists
    persist_dir = "chroma_20newsgroups"
    os.makedirs(persist_dir, exist_ok=True)

    # Create persistent Chroma client
    # Use Chroma as a lightweight, embedded vector DB.
    # - PersistentClient: store the index on disk so later phases (GMM, cache, FastAPI)
    #   can reuse it without recomputing embeddings.
    # - hnsw:space='cosine': match the cosine similarity notion used by sentence embeddings.
    # - Metadata stores label + short snippet for filtered retrieval and debugging.
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )

    # Create or get collection
    collection = client.get_or_create_collection(
        name="twenty_newsgroups",
        metadata={"hnsw:space": "cosine"},
    )


    # Convert numpy arrays to plain Python types
    ids_list = ids.astype(str).tolist()
    embeddings_list = embeddings.astype("float32").tolist()
    metadatas = [
        {  
             # label_idx and label_name allow downstream components to:
            # - filter search to specific topics,
            # - evaluate cluster quality per true label.
            "label_idx": int(idx),
            "label_name": str(name),
            # Store only a short snippet in metadata for previews.
            # Full texts stay in artifacts on disk; this keeps the vector DB lean.
            "snippet": str(text)[:200],
        }
        for idx, name, text in zip(label_indices, label_names, texts_clean)
    ]

    # Add to collection in batches
    batch_size = 512
    n = len(ids_list)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_ids = ids_list[start:end]
        batch_embeddings = embeddings_list[start:end]
        batch_metadatas = metadatas[start:end]
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
        )
        print(f"Inserted {end} / {n} vectors")

    print(f"Chroma collection '{collection.name}' is ready and persisted at '{persist_dir}'")


if __name__ == "__main__":
    main()
