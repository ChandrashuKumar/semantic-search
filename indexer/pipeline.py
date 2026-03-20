# indexer/pipeline.py

import yaml

from indexer.crawler import Crawler
from indexer.extractor import Extractor
from indexer.chunker import Chunker
from indexer.embedder import Embedder
from indexer.store import Store


class IndexingPipeline:
    """
    Wires all indexer modules together.

    The flow for each file:
        Crawler (discover + hash check)
            → Extractor (file → raw text)
                → Chunker (text → chunks with metadata)
                    → Embedder (chunks → vectors)
                        → Store (vectors → FAISS, metadata → SQLite)
    """

    def __init__(self, config_path="config.yaml"):
        """
        Initialize all pipeline components.
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.crawler = Crawler(config_path)
        self.extractor = Extractor()
        self.chunker = Chunker(
            chunk_size=config.get("chunk_size", 500),
            overlap=config.get("overlap", 50),
        )
        self.embedder = Embedder(config_path)
        self.store = Store(config_path)

    def run(self):
        """
        Execute the full indexing pipeline.
        """
        known_hashes = self.store.load_hashes()
        print("Scanning for new/modified files...")
        files_to_process, current_hashes, deleted_files = self.crawler.get_new_and_modified(known_hashes)

        for filepath in deleted_files:
            self.store.remove_file_chunks(filepath)

        if not files_to_process:
            print("Index is up to date.")
            print(f"Total vectors: {self.store.get_total_vectors()}")
            return

        total = len(files_to_process)
        for i, filepath in enumerate(files_to_process, 1):
            print(f"[{i}/{total}] {filepath}")
            text = self.extractor.extract(filepath)
            if not text.strip():
                print(f"  Skipping (no text extracted)")
                continue
            chunks = self.chunker.chunk_file(text, filepath)
            chunk_texts = [c["text"] for c in chunks]
            embeddings = self.embedder.embed_chunks(chunk_texts)
            self.store.remove_file_chunks(filepath)
            self.store.add_chunks(chunks, embeddings)
            self.store.save_file_info(filepath, current_hashes[filepath], len(chunks))

        print(f"\nProcessed {total} files.")
        print(f"Total vectors: {self.store.get_total_vectors()}")


# --- Test ---
if __name__ == "__main__":
    pipeline = IndexingPipeline()
    pipeline.run()