# search/pipeline.py

from search.expander import QueryExpander
from search.dense import DenseSearch
from search.sparse import SparseSearch
from search.fusion import RRFFusion
from search.reranker import Reranker


class SearchPipeline:
    """
    Wires all search modules together.
    
    Flow:
        User query
            → QueryExpander (add synonyms)
            → DenseSearch (FAISS, gets expanded query)
            → SparseSearch (BM25, gets original query)
            → RRFFusion (merge both result lists)
            → Reranker (cross-encoder precision pass)
            → Final results
    """

    def __init__(self, config_path="config.yaml"):
        """
        Initialize all search components.
        """
        self.query_expander = QueryExpander(max_synonyms_per_word=3)
        self.dense = DenseSearch(config_path)
        self.sparse = SparseSearch(config_path)
        self.fusion = RRFFusion(k=60)
        self.reranker = Reranker(config_path)

    def search(self, query, top_k=10, fusion_k=20):
        """
        Execute the full search pipeline.

        Args:
            query (str) — the user's raw search query
            top_k (int) — how many final results to return
            fusion_k (int) — how many results to get from each
                             search method before fusion (default 20)

        Returns:
            list[dict] — final ranked results
        """
        expanded = self.query_expander.expand_query(query)
        dense_results = self.dense.search(expanded, top_k=fusion_k)
        sparse_results = self.sparse.search(query, top_k=fusion_k)
        fused = self.fusion.fuse(dense_results, sparse_results)
        reranked = self.reranker.rerank(query, fused, top_k=top_k)
        return reranked


# --- Test ---
if __name__ == "__main__":
    pipeline = SearchPipeline()

    queries = ["manchester", "frequency modulation", "nyquist theorem"]

    for q in queries:
        print(f"Query: {q}")
        results = pipeline.search(q, top_k=3)
        for r in results:
            print(f"  [{r['rerank_score']:.4f}] {r['filepath']}")
            print(f"          {r['chunk_text'][:80]}...")
        print()