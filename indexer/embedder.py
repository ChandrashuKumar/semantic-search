# indexer/embedder.py

import yaml
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Loads a sentence-transformer model and converts text chunks
    into dense vector embeddings.
    """

    def __init__(self, config_path="config.yaml"):
        """
        Load the config and initialize the embedding model.

        Args:
            config_path (str) — path to config.yaml
        """
        with open(config_path, "r") as f:
            config=yaml.safe_load(f)
        
        model_name = config["embedding_model"]
        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(model_name)

    def embed_chunks(self, chunks):
        """
        Convert a list of text chunks into dense vector embeddings.

        Args:
            chunks (list[str]) — list of text strings to embed

        Returns:
            list[list[float]] — list of embedding vectors
                                each vector is 384 floats for MiniLM
        """
        embeddings = self.model.encode(chunks)
        return embeddings

    def embed_single(self, text):
        """
        Embed a single text string (for a single query).

        Args:
            text (str) — a single text string

        Returns:
            list[float] — one embedding vector
        """
        return self.model.encode(text)


# --- Test ---
if __name__ == "__main__":
    embedder = Embedder()

    test_chunks = [
        "The quarterly budget report shows increased spending",
        "Machine learning models can understand text semantics",
        "The cat sat on the mat and looked out the window"
    ]

    print("Embedding 3 test chunks...")
    vectors = embedder.embed_chunks(test_chunks)
    print(f"Got {len(vectors)} vectors")
    print(f"Each vector has {len(vectors[0])} dimensions")
    print(f"First vector (first 5 values): {vectors[0][:5]}")

    print("\n--- Single query embedding ---")
    query_vec = embedder.embed_single("budget spending report")
    print(f"Query vector: {len(query_vec)} dimensions")