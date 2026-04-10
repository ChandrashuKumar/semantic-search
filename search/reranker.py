# search/reranker.py

import yaml
from sentence_transformers import CrossEncoder


class Reranker:
    """
    Re-ranks search results using a cross-encoder model.
    """

    def __init__(self, config_path="config.yaml", model=None):
        """
        Load the cross-encoder model, or accept a pre-loaded one.
        """
        if model is not None:
            self.model = model
        else:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            self.model = CrossEncoder(config["reranker_model"])

    def rerank(self, query, results, top_k=10):
        """
        Re-rank a list of search results using the cross-encoder.

        Args:
            query (str) — the original user query
            results (list[dict]) — fused results from RRFFusion.fuse()
                Each has: chunk_id, filepath, chunk_text, chunk_index, rrf_score
            top_k (int) — how many final results to return

        Returns:
            list[dict] — re-ranked results, each containing:
                {
                    "chunk_id": 42,
                    "filepath": "/docs/report.pdf",
                    "chunk_text": "...",
                    "chunk_index": 3,
                    "rrf_score": 0.032,      # original RRF score
                    "rerank_score": 0.95,    # cross-encoder score
                    "dense_rank": 3,
                    "sparse_rank": 1,
                }
        """
        if len(results) == 0:
            return []
        
        pairs = [[query, r["chunk_text"]] for r in results]
        scores = self.model.predict(pairs)
        for i, result in enumerate(results):
            result["rerank_score"] = float(scores[i])
            
        results.sort(key=lambda r: r["rerank_score"], reverse=True)
        return results[:top_k]


# --- Test ---
if __name__ == "__main__":
    reranker = Reranker()

    query = "manchester encoding"

    # Simulate RRF results
    fake_results = [
        {"chunk_id": 3, "filepath": "digital.pdf", "chunk_text": "Manchester encoding is a type of digital encoding in which each bit is represented by a transition", "chunk_index": 0, "rrf_score": 0.032, "dense_rank": 3, "sparse_rank": 1},
        {"chunk_id": 2, "filepath": "nwmodels.pdf", "chunk_text": "The OSI model has seven layers that define network communication", "chunk_index": 0, "rrf_score": 0.031, "dense_rank": 2, "sparse_rank": 3},
        {"chunk_id": 1, "filepath": "hello.txt", "chunk_text": "hello world this is a test file", "chunk_index": 0, "rrf_score": 0.016, "dense_rank": 1, "sparse_rank": None},
    ]

    results = reranker.rerank(query, fake_results, top_k=3)
    print(f"Query: {query}\n")
    for r in results:
        print(f"  Rerank: {r['rerank_score']:.4f} | RRF: {r['rrf_score']:.5f} | {r['filepath']}")
        print(f"          {r['chunk_text'][:80]}...")
        print()