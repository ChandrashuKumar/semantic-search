# indexer/chunker.py


class Chunker:
    """
    Splits extracted text into overlapping chunks using a sliding window.
    Each chunk will later be embedded as a separate vector.
    
    Why chunk at all?
    - Embedding models have a token limit (typically 256-512 tokens)
    - A 50-page PDF as one embedding would lose detail
    - Small chunks let us pinpoint the EXACT passage that matches a query
    
    Why overlap?
    - A sentence at the boundary might get cut in half
    - Overlap ensures every sentence appears fully in at least one chunk
    """

    def __init__(self, chunk_size=500, overlap=50):
        """
        Args:
            chunk_size (int) — max number of words per chunk
            overlap (int) — number of words shared between consecutive chunks

        TODO:
        - Store chunk_size and overlap as instance variables
        - Validate that overlap is less than chunk_size
            (if overlap >= chunk_size, chunks would never advance forward)
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

        Example with chunk_size=5, overlap=2:
            text = "The quick brown fox jumps over the lazy dog today"
            words = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "today"]

            Chunk 0: words[0:5]  → "The quick brown fox jumps"
            Chunk 1: words[3:8]  → "fox jumps over the lazy"      (step = 5-2 = 3)
            Chunk 2: words[6:11] → "the lazy dog today"           (step = 3 again)

        TODO:
        - Split the text into a list of words using .split()
        - If the word list is empty, return an empty list
        - Calculate step size: step = chunk_size - overlap
        - Use a loop starting at 0, stepping by 'step', up to len(words)
        - At each position, take words[i : i + chunk_size]
        - Join each slice back into a string with " ".join()
        - Return the list of chunk strings

        HINT:
            words = text.split()
            step = self.chunk_size - self.overlap
            for i in range(0, len(words), step):
                chunk_words = words[i : i + self.chunk_size]
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
        This metadata will be stored in SQLite alongside the vectors.

        Args:
            text (str) — extracted text content
            filepath (str) — source file path (for metadata)

        Returns:
            list[dict] — each dict contains:
                {
                    "text": "the chunk text...",
                    "filepath": "/path/to/file.pdf",
                    "chunk_index": 0,     # position in the file
                    "total_chunks": 5     # how many chunks this file produced
                }

        TODO:
        - Call self.chunk_text(text) to get the list of chunk strings
        - Build a list of dicts, one per chunk, with the fields shown above
        - chunk_index starts at 0

        HINT:
            chunks = self.chunk_text(text)
            for i, chunk in enumerate(chunks):
                # build the dict here
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