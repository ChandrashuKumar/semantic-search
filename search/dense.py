# search/dense.py

import numpy as np
from indexer.embedder import Embedder
from indexer.store import Store
import sqlite3


class DenseSearch:
    """
    Searches the FAISS vector index for chunks that are semantically
    similar to the query.
    
    FAISS returns two arrays:
        - distances: how far each result is from the query (lower = more similar)
        - indices: the vector IDs of the closest matches
    """

    def __init__(self, config_path="config.yaml"):
        """
        Initialize the Embedder and Store modules.
        """
        self.embedder = Embedder(config_path)
        self.store = Store(config_path)

    def search(self, query, top_k=20):
        """
        Search FAISS for the most semantically similar chunks.

        Args:
            query (str) — the (expanded) search query
            top_k (int) — how many results to return

        Returns:
            list[dict] — ranked results, each containing:
                {
                    "chunk_id": 42,
                    "filepath": "/docs/report.pdf",
                    "chunk_text": "the actual chunk content...",
                    "chunk_index": 3,
                    "score": 0.85,     # similarity score 
                    "source": "dense"  
                }
        """
        if self.store.index is None:
            return []

        query_vector = self.embedder.embed_single(query).reshape(1, -1).astype("float32")

        distances, indices = self.store.index.search(query_vector, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            chunk_metadata = self._get_chunk_metadata(idx)
            if chunk_metadata is None:
                continue

            similarity_score = 1 / (1 + dist)
            results.append({
                "chunk_id": idx,
                "filepath": chunk_metadata["filepath"],
                "chunk_text": chunk_metadata["chunk_text"],
                "chunk_index": chunk_metadata["chunk_index"],
                "score": similarity_score,
                "source": "dense"
            })

        return results

    def _get_chunk_metadata(self, chunk_id):
        """
        Look up a chunk's metadata from SQLite by its ID.

        Args:
            chunk_id (int) — the FAISS vector ID (matches chunks.id)

        Returns:
            dict or None — {"filepath": ..., "chunk_text": ..., "chunk_index": ...}
        """
        conn = sqlite3.connect(self.store.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT filepath, chunk_text, chunk_index FROM chunks WHERE id = ?", (int(chunk_id),))
        result = cursor.fetchone()
        conn.close()

        return {
        "filepath": result[0],
        "chunk_text": result[1],
        "chunk_index": result[2]
        }


# --- Test ---
if __name__ == "__main__":
    searcher = DenseSearch()

    test_queries = [
        "manchester",
        "frequency modulation",
        "nyquist theorem",
    ]

    for q in test_queries:
        print(f"Query: {q}")
        results = searcher.search(q, top_k=3)
        for r in results:
            print(f"  [{r['score']:.3f}] {r['filepath']}")
            print(f"          {r['chunk_text'][:80]}...")
        print()