# indexer/store.py

import os
import sqlite3
import numpy as np
import faiss
import yaml


class Store:
    """
    Handles two storage systems:
    
    1. FAISS — stores dense vectors for fast similarity search
    
    2. SQLite — stores metadata about each chunk
    
    Both are stored as files on disk inside data_dir.
    """

    def __init__(self, config_path="config.yaml"):
        """
        Load config, set up file paths, initialize FAISS index and SQLite database.
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        self.data_dir = config["data_dir"]
        os.makedirs(self.data_dir, exist_ok=True)

        self.faiss_path = os.path.join(self.data_dir, "index.faiss")
        self.db_path = os.path.join(self.data_dir, "metadata.db")

        self._init_db()
        self._load_or_create_index()

    def _init_db(self):
        """
        Create SQLite tables if they don't already exist.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                filepath TEXT NOT NULL,
                chunk_text TEXT NOT NULL,
                chunk_index INTEGER,
                FOREIGN KEY (filepath) REFERENCES files(filepath)
            )
        ''')        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS files (
                filepath TEXT PRIMARY KEY,
                mtime REAL NOT NULL,
                size INTEGER NOT NULL,
                total_chunks INTEGER
            )
        ''')
        conn.commit()
        conn.close()

    def _load_or_create_index(self):
        """
        Load an existing FAISS index from disk, or create a new empty one.
        """
        if os.path.exists(self.faiss_path):
            self.index = faiss.read_index(self.faiss_path)
        else:
            self.index = None

    def get_next_id(self):
        """Get the next available chunkID from sqlite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(id) FROM chunks")
        result = cursor.fetchone()[0]
        conn.close()
        if result is None:
            return 0
        return result + 1

    def add_chunks(self, chunks_with_metadata, embeddings):
        """
        Add new chunks and their embeddings to both FAISS and SQLite.

        Args:
            chunks_with_metadata (list[dict]) — from chunker.chunk_file()
                Each dict has: text, filepath, chunk_index
            embeddings (numpy.ndarray) — shape (num_chunks, embedding_dim)
                From embedder.embed_chunks()
        """

        #add embeddings to faiss
        embeddings = embeddings.astype("float32")
        if self.index is None:
            dimension = embeddings.shape[1]
            base_index = faiss.IndexFlatL2(dimension)   #L2 Euclidean distance
            self.index = faiss.IndexIDMap2(base_index)  #map custom IDS for deletion purpose

        start_id  = self.get_next_id()
        ids = np.array([start_id + i for i in range(len(chunks_with_metadata))], dtype=np.int64)

        self.index.add_with_ids(embeddings,ids)
        faiss.write_index(self.index, self.faiss_path)

        #add metadata to sqlite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for i, chunk in enumerate(chunks_with_metadata):
            vector_id = start_id + i
            cursor.execute(
                "INSERT INTO chunks (id, filepath, chunk_text, chunk_index) VALUES (?, ?, ?, ?)",
                (vector_id, chunk["filepath"], chunk["text"], chunk["chunk_index"])
            )

        conn.commit()
        conn.close()
        

    def save_file_info(self, filepath, mtime, size, total_chunks):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO files (filepath, mtime, size, total_chunks) VALUES (?, ?, ?, ?)",
            (filepath, mtime, size, total_chunks)
        )
        conn.commit()
        conn.close()
        

    def load_file_info(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT filepath, mtime, size FROM files")
        rows = cursor.fetchall()
        conn.close()
        return {row[0]: {"mtime": row[1], "size": row[2]} for row in rows}
    
    
    def remove_file_chunks(self, filepath):
        """Delete all chunks belonging to a file from SQLite, along with their embeddings from faiss."""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        ids = cursor.execute("SELECT id FROM chunks WHERE filepath = ?", (filepath,)).fetchall()
        cursor.execute("DELETE FROM chunks WHERE filepath = ?", (filepath,))
        cursor.execute("DELETE FROM files WHERE filepath = ?", (filepath,))
        conn.commit()
        conn.close()

        if ids and self.index is not None:
            self.index.remove_ids(np.array([i[0] for i in ids], dtype=np.int64))
            faiss.write_index(self.index, self.faiss_path)

    def get_total_vectors(self):
        """
        Return how many vectors are currently in the FAISS index.

        Returns:
            int — number of vectors, or 0 if index doesn't exist yet
        """
        if self.index is None:
            return 0
        else:
            return self.index.ntotal


# --- Test ---
if __name__ == "__main__":
    import numpy as np

    store = Store()

    # Simulate some chunks from the chunker
    fake_chunks = [
        {"text": "quarterly budget report summary", "filepath": "/docs/report.pdf", "chunk_index": 0, "total_chunks": 2},
        {"text": "revenue increased by fifteen percent", "filepath": "/docs/report.pdf", "chunk_index": 1, "total_chunks": 2},
        {"text": "python machine learning tutorial", "filepath": "/docs/tutorial.txt", "chunk_index": 0, "total_chunks": 1},
    ]

    # Simulate embeddings (3 chunks × 384 dimensions)
    fake_embeddings = np.random.rand(3, 384).astype("float32")

    print(f"Vectors before: {store.get_total_vectors()}")
    store.add_chunks(fake_chunks, fake_embeddings)
    print(f"Vectors after: {store.get_total_vectors()}")
