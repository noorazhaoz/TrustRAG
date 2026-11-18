
"""Trace Q4 reference evidence through retrieval stages."""

import numpy as np

from src.adaptive_retriever import AdaptiveRetriever
from src.document_store import DocumentStore

# Import BM25 and rrf from the same module used
# by src/adaptive_retriever.py.
from src.adaptive_retriever import BM25, rrf


DOCUMENT_ID = "paper-demo-001"
QUESTION = "How does the method combine two pre-trained models?"
TARGET_ID = "page13_chunk3"


def rank_of(order, target):
    positions = np.flatnonzero(
        np.asarray(order) == target
    )
    return (
        int(positions[0]) + 1
        if len(positions) else None
    )


def main():
    retriever = AdaptiveRetriever(router=None)
    DocumentStore().restore_all(retriever)

    doc = retriever.documents[DOCUMENT_ID]
    vectors = retriever.vectors[DOCUMENT_ID]

    ids = [str(u["paragraph_id"]) for u in doc]
    target = ids.index(TARGET_ID)

    qvec = retriever.embedder.encode(
        [QUESTION],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")[0]

    dense_scores = vectors @ qvec

    dense_order = np.argsort(
        -dense_scores,
        kind="stable",
    )

    bm25 = BM25([u["text"] for u in doc])
    bm_scores = bm25.score(QUESTION)
    bm_order = np.argsort(
        -bm_scores,
        kind="stable",
    )

    hybrid_order = rrf(
        dense_order,
        bm_order,
        len(doc),
        constant=60,
    )

    keep_top10 = retriever._rerank(
        QUESTION, doc, dense_order[:30]
    )

    hybrid_top10 = retriever._rerank(
        QUESTION, doc, hybrid_order[:30]
    )

    print("\nTarget:", TARGET_ID)
    print("Dense rank:", rank_of(dense_order, target))
    print("BM25 rank:", rank_of(bm_order, target))
    print("RRF rank:", rank_of(hybrid_order, target))

    print("\nCandidate inclusion:")
    print("Dense Top-30:", target in dense_order[:30])
    print("Hybrid Top-30:", target in hybrid_order[:30])

    print("\nFinal evidence:")
    print("Keep Top-10:", target in keep_top10)
    print("Hybrid Top-10:", target in hybrid_top10)

    # Examine the complete reranked candidate lists.
    keep_full = retriever._rerank(
        QUESTION, doc, dense_order[:30]
    )

    hybrid_full = retriever._rerank(
        QUESTION, doc, hybrid_order[:30]
    )

    print("\nCross-Encoder ranking:")
    print("Keep:", rank_of(keep_full, target))
    print("Hybrid:", rank_of(hybrid_full, target))



    # Full Cross-Encoder rankings (before Top-10 truncation).
    def inspect_reranking(name, candidates):
        order = [int(i) for i in candidates]

        scores = np.asarray(
            retriever.cross_encoder.predict(
                [(QUESTION, doc[i]["text"]) for i in order],
                show_progress_bar=False,
            )
        ).reshape(-1)

        if len(scores) != len(order):
            raise ValueError("Cross-encoder score count mismatch")

        if not np.isfinite(scores).all():
            raise ValueError("Non-finite cross-encoder scores")

        score_map = {
            i: float(score)
            for i, score in zip(order, scores)
        }

        # Same tie-breaking rule as production _rerank().
        full_order = sorted(
            order,
            key=lambda i: (-score_map[i], i),
        )

        target_rank = rank_of(full_order, target)

        print(f"\n{name} Cross-Encoder:")
        print("Target rank:", target_rank)
        print("Target score:", round(score_map[target], 6))
        print("Top-10 cutoff score:", round(
            score_map[full_order[9]], 6
        ))

        print("\nNearby candidates:")
        if target_rank is not None:
            start = max(0, target_rank - 3)
            end = min(len(full_order), target_rank + 2)

            for position in range(start, end):
                idx = full_order[position]
                print(
                    f"  Rank {position + 1}: "
                    f"{doc[idx]['paragraph_id']} "
                    f"(score={score_map[idx]:.6f})"
                )

    inspect_reranking("KEEP", dense_order[:30])
    inspect_reranking("HYBRID", hybrid_order[:30])

    print("\nDense Raw Top-10:")
    for rank, idx in enumerate(dense_order[:10], 1):
        print(f"{rank:2d}. {doc[int(idx)]['paragraph_id']}")



if __name__ == "__main__":
    main()
