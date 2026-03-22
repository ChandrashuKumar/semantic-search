# indexer/chunker.py


class Chunker:
    """
    Splits extracted text into overlapping chunks using a sliding window.
    """

    def __init__(self, chunk_size=500, overlap=50):
        """
        Args:
            chunk_size (int) — max number of words per chunk
            overlap (int) — number of words shared between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        if self.overlap >= self.chunk_size:
            raise ValueError("Overlap must be smaller than chunk_size")

    def chunk_text(self, text):
        """
        Split a text string into overlapping chunks based on word count.

        Args:
            text (str) — the full extracted text from a file

        Returns:
            list[str] — list of text chunks
        """
        words = text.split()
        if not words:
            return []
        step = self.chunk_size - self.overlap
        chunks = []
        for i in range(0, len(words), step):
            chunk_words = words[i:i+self.chunk_size]
            chunks.append(" ".join(chunk_words))
        return chunks

    def chunk_file(self, text, filepath):
        """
        Chunk a file's text and attach metadata to each chunk.

        Args:
            text (str) — extracted text content
            filepath (str) — source file path (for metadata)

        Returns:
            list[dict] — each dict contains:
                {
                    "text": "the chunk text...",
                    "filepath": "/path/to/file.pdf",
                    "chunk_index": 0,     # position in the file
                }
        """
        chunks = self.chunk_text(text)
        results = []
        for i, chunk in enumerate(chunks):
            results.append({
                "text": chunk,
                "filepath": filepath,
                "chunk_index": i,
            })
        return results


# --- Test ---
if __name__ == "__main__":
    chunker = Chunker(chunk_size=10, overlap=3)

    sample = (
        "The quick brown fox jumps over the lazy dog. "
        "Semantic search finds files by meaning not just keywords. "
        "This is a test of the chunking system for our project."
    )

    chunks = chunker.chunk_text(sample)
    print(f"Text has {len(sample.split())} words → {len(chunks)} chunks\n")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {chunk}")

    print("\n--- With metadata ---")
    results = chunker.chunk_file(sample, "/test/sample.txt")
    for r in results:
        print(f"[{r['chunk_index']}] {r['text'][:60]}...")