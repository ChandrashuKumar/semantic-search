# search/expander.py
import nltk

#download wordnet data
try:
    from nltk.corpus import wordnet
    wordnet.synsets("test")  # triggers error if data not downloaded
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    from nltk.corpus import wordnet



class QueryExpander:
    """
    Expands a user's search query with synonyms and related words
    using WordNet — an English dictionary/thesaurus built into NLTK.
    
    WordNet organizes words into "synsets" (synonym sets).
    Each synset is a group of words that mean the same thing:
        car → {car, auto, automobile, motorcar, machine}
    """

    def __init__(self, max_synonyms_per_word=3):
        """
        Args:
            max_synonyms_per_word (int) 
        """
        self.max_synonyms_per_word = max_synonyms_per_word

    def get_synonyms(self, word):
        """
        Get synonyms for a single word from WordNet.

        Args:
            word (str) — a single word

        Returns:
            set[str] — set of synonyms (excluding the original word)
        """
        synsets = wordnet.synsets(word)
        synonyms = set()
        
        for syn in synsets:
            for lemma in syn.lemma_names():
                if lemma != word:
                    synonyms.add(lemma.replace("_", " "))
                if len(synonyms) == self.max_synonyms_per_word:
                    return synonyms
                
        return synonyms

    def expand_query(self, query):
        """
        Expand a full query string by adding synonyms for each word.

        Args:
            query (str) — the user's original search query

        Returns:
            str — expanded query with synonyms appended
        """
        words = query.lower().split()
        all_synonyms = []
        for word in words:
            synonyms = self.get_synonyms(word)
            all_synonyms.extend(synonyms)
        expanded = query + " " + " ".join(all_synonyms)

        return expanded


# --- Test ---
if __name__ == "__main__":
    expander = QueryExpander(max_synonyms_per_word=3)

    test_queries = [
        "budget report",
        "machine learning",
        "fast car",
        "happy dog",
    ]

    for q in test_queries:
        expanded = expander.expand_query(q)
        print(f"Original:  {q}")
        print(f"Expanded:  {expanded}")
        print()