# search/fusion.py


class RRFFusion:
    """
    Combines results from dense (FAISS) and sparse (BM25) search
    using Reciprocal Rank Fusion.
    """

    def __init__(self, k=60):
        """
        Args:
            k (int) — smoothing constant
                Higher k = less difference between adjacent ranks
        """
        self.k = k

    def fuse(self, dense_results, sparse_results):
        """
        Merge two ranked result lists using RRF.

        Args:
            dense_results (list[dict]) — from DenseSearch.search()
            sparse_results (list[dict]) — from SparseSearch.search()
            Both have: chunk_id, filepath, chunk_text, chunk_index, score, source

        Returns:
            list[dict] — fused results sorted by RRF score descending
                Each dict contains:
                {
                    "chunk_id": 42,
                    "filepath": "/docs/report.pdf",
                    "chunk_text": "...",
                    "chunk_index": 3,
                    "rrf_score": 0.032,
                    "dense_rank": 3,     
                    "sparse_rank": 1,     
                }
        """
        fused = {}

        for rank, result in enumerate(dense_results, start=1):
            rrf_score = 1 / (self.k + rank)

            chunk_id = result["chunk_id"]

            fused[chunk_id] = {
                "chunk_id": chunk_id,
                "filepath": result["filepath"],
                "chunk_text": result["chunk_text"],
                "chunk_index": result["chunk_index"],
                "rrf_score": rrf_score,
                "dense_rank": rank,
                "sparse_rank": None,
            }

        for rank, result in enumerate(sparse_results, start=1):
            rrf_score = 1 / (self.k + rank)

            chunk_id = result["chunk_id"]

            if chunk_id in fused:
                fused[chunk_id]["rrf_score"] += rrf_score
                fused[chunk_id]["sparse_rank"] = rank
            else:
                fused[chunk_id] = {
                    "chunk_id": chunk_id,
                    "filepath": result["filepath"],
                    "chunk_text": result["chunk_text"],
                    "chunk_index": result["chunk_index"],
                    "rrf_score": rrf_score,
                    "dense_rank": None,
                    "sparse_rank": rank,
                }

        fused = list(fused.values())
        fused.sort(key=lambda x: x["rrf_score"], reverse=True)

        return fused


# --- Test ---
if __name__ == "__main__":
    fusioner = RRFFusion(k=60)

    # Simulate dense results
    dense = [
        {"chunk_id": 1, "filepath": "hello.txt", "chunk_text": "hello world", "chunk_index": 0, "score": 0.389, "source": "dense"},
        {"chunk_id": 2, "filepath": "nwmodels.pdf", "chunk_text": "network models...", "chunk_index": 0, "score": 0.355, "source": "dense"},
        {"chunk_id": 3, "filepath": "digital.pdf", "chunk_text": "manchester encoding...", "chunk_index": 0, "score": 0.350, "source": "dense"},
    ]

    # Simulate sparse results
    sparse = [
        {"chunk_id": 3, "filepath": "digital.pdf", "chunk_text": "manchester encoding...", "chunk_index": 0, "score": 2.632, "source": "sparse"},
        {"chunk_id": 4, "filepath": "data_signal.pdf", "chunk_text": "nyquist theorem...", "chunk_index": 0, "score": 2.136, "source": "sparse"},
        {"chunk_id": 2, "filepath": "nwmodels.pdf", "chunk_text": "network models...", "chunk_index": 0, "score": 1.500, "source": "sparse"},
    ]

    results = fusioner.fuse(dense, sparse)
    print("Fused results for 'manchester':\n")
    for r in results:
        print(f"  RRF: {r['rrf_score']:.5f} | Dense rank: {r['dense_rank']} | Sparse rank: {r['sparse_rank']} | {r['filepath']}")