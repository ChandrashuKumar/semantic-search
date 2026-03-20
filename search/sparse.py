# search/sparse.py

import sqlite3
import yaml
from rank_bm25 import BM25Okapi
import os


class SparseSearch:
    """
    Keyword-based search using BM25 (Best Matching 25).
    
    BM25 works on tokenized text — we split each chunk into words
    and build an index over all chunks.
    """

    def __init__(self, config_path="config.yaml"):
        """
        Load all chunk texts from SQLite and build a BM25 index.
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.db_path = os.path.join(config["data_dir"], "metadata.db")
        self._build_index()

    def _build_index(self):
        """
        Load all chunks from SQLite and build a BM25 index.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, filepath, chunk_text, chunk_index FROM chunks")
        rows = cursor.fetchall()
        self.chunks = [
            {"id": row[0], "filepath": row[1], "chunk_text": row[2], "chunk_index": row[3]}
            for row in rows
        ]
        if len(self.chunks) == 0:
            self.bm25 = None
        else:
            tokenized = [chunk["chunk_text"].lower().split() for chunk in self.chunks]
            self.bm25 = BM25Okapi(tokenized)
        conn.close()

    def search(self, query, top_k=20):
        """
        Search for chunks matching the query using BM25 keyword scoring.

        Args:
            query (str) — the search query (original, NOT expanded)
            top_k (int) — how many results to return

        Returns:
            list[dict] — ranked results, each containing:
                {
                    "chunk_id": 42,
                    "filepath": "/docs/report.pdf",
                    "chunk_text": "the actual chunk content...",
                    "chunk_index": 3,
                    "score": 2.45,     # BM25 score 
                    "source": "sparse" 
                }

        NOTE:
            BM25 scores can be any positive number.
            A score of 0 means no matching terms at all.
        """
        
        if self.bm25 == None:
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for i in top_indices:
            if scores[i] > 0:
                results.append({
                    "chunk_id": self.chunks[i]["id"],
                    "filepath": self.chunks[i]["filepath"],
                    "chunk_text": self.chunks[i]["chunk_text"],
                    "chunk_index": self.chunks[i]["chunk_index"],
                    "score": scores[i],
                    "source": "sparse"
                })
        return results


# --- Test it ---
if __name__ == "__main__":
    searcher = SparseSearch()

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
            print(f"          {r['chunk_text'][:250]}...")
        print()