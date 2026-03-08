# cache_debug.py
from sentence_transformers import SentenceTransformer
import joblib
import numpy as np

from semantic_cache import SemanticCache, cosine_similarity


def print_pairwise_sims(embedder):
    """Optional: inspect similarities of paraphrase pairs."""
    pairs = [
        (
            "NASA space shuttle mission details",
            "space shuttle launch by NASA",
        ),
        (
            "machine learning algorithms comparison",
            "comparing different ML algorithms",
        ),
        (
            "python programming tutorial",
            "how to learn python coding",
        ),
    ]
    print("\nPairwise cosine similarities:")
    for a, b in pairs:
        ea = embedder.encode([a])[0]
        eb = embedder.encode([b])[0]
        sim = cosine_similarity(ea, eb)
        print(f"  '{a[:25]}...' vs '{b[:25]}...' -> {sim:.3f}")


def run_config(name, threshold, allow_cross_cluster, gmm, embedder, test_queries):
    print(f"\n=== Config: {name} ===")
    print(
        f"  similarity_threshold={threshold}, "
        f"allow_cross_cluster={allow_cross_cluster}"
    )

    cache = SemanticCache(
        gmm_model=gmm,
        embedder=embedder,
        similarity_threshold=threshold,
        allow_cross_cluster=allow_cross_cluster,
    )

    for q in test_queries:
        result, meta = cache.get(q)
        is_hit = bool(meta.get("hit", meta.get("cache_hit", False)))

        if not is_hit:
            # simulate expensive computation
            dummy_result = f"Top docs for: {q}"
            cache.add(q, dummy_result)
            print(f"MISS | '{q}'")
        else:
            sim = meta.get("similarity_score")
            sim_str = f"{sim:.3f}" if sim is not None else "n/a"
            matched_q = meta.get("matched_query", "")
            reason = meta.get("reason", "")
            print(
                f"HIT  | '{q}' "
                f"-> reused result for '{matched_q}' "
                f"(sim={sim_str}, reason={reason})"
            )

    stats = cache.get_stats()
    print(
        f"Stats: hits={stats['hit_count']}, "
        f"misses={stats['miss_count']}, "
        f"hit_rate={stats['hit_rate']:.1%}, "
        f"entries={stats['total_entries']}"
    )



def main():
    # ---------- load models ----------
    print("Loading models...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    gmm = joblib.load("artifacts/gmm_diag20_model.pkl")

    # ---------- demo queries ----------
    test_queries = [
        "NASA space shuttle mission details",
        "space shuttle launch by NASA",  # paraphrase
        "machine learning algorithms comparison",
        "comparing different ML algorithms",  # paraphrase
        "python programming tutorial",
        "how to learn python coding",  # paraphrase
    ]

    print("\n=== Semantic Cache Demo ===")

    # Optional: show raw cosine similarities between paraphrase pairs
    print_pairwise_sims(embedder)

    # Conservative: strict, no cross‑cluster
    run_config(
        name="Conservative (strict)",
        threshold=0.85,
        allow_cross_cluster=False,
        gmm=gmm,
        embedder=embedder,
        test_queries=test_queries,
    )

    # Balanced: slightly lower threshold, cross‑cluster enabled
    run_config(
        name="Balanced",
        threshold=0.78,
        allow_cross_cluster=True,
        gmm=gmm,
        embedder=embedder,
        test_queries=test_queries,
    )

    # Aggressive: low threshold, cross‑cluster enabled
    run_config(
        name="Aggressive (high recall)",
        threshold=0.70,
        allow_cross_cluster=True,
        gmm=gmm,
        embedder=embedder,
        test_queries=test_queries,
    )


if __name__ == "__main__":
    main()
