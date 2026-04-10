# main.py
import os
import threading
import yaml
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
# os.environ["HF_HUB_VERBOSITY"] = "error"
# os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


from indexer.pipeline import IndexingPipeline
from indexer.watcher import Watcher
from search.pipeline import SearchPipeline
from watchdog.observers import Observer


def start_watcher_background(watcher):
    """
    Start the watchdog observer in a background thread.

    Args:
        watcher (Watcher) — initialized watcher instance

    Returns:
        Observer — the running observer 
    """
    observer = Observer()
    for path in watcher.watch_paths:
        observer.schedule(watcher.handler, path, recursive=True)
    observer.start()
    return observer


def search_loop(pipeline):
    """
    Interactive search loop.

    Args:
        pipeline (SearchPipeline) — initialized search pipeline
    """
    while(True):
        query = input("\nSearch: ").strip()

        if(query == ""):
            continue

        if query.lower() in ("exit", "quit"):
            break
        
        query_results = pipeline.search(query, top_k=5)

        if len(query_results) == 0:
            print("No results found.")
        else:
            for idx, result in enumerate(query_results):
                print(f"{idx+1}. {result['filepath']}")
                print(f"  [{result['rerank_score']:.4f}] {result['chunk_text'][:150]}...")
                print()


def main():
    """
    Main entry point.
    """
    observer = None

    try:
        with open("config.yaml") as f:
            config = yaml.safe_load(f)

        reranker_model = None

        def _load_reranker():
            nonlocal reranker_model
            from sentence_transformers import CrossEncoder
            try:
                reranker_model = CrossEncoder(config["reranker_model"], local_files_only=True)
            except Exception:
                reranker_model = CrossEncoder(config["reranker_model"])

        reranker_thread = threading.Thread(target=_load_reranker, daemon=True)
        reranker_thread.start()

        indexing_pipeline = IndexingPipeline()
        indexing_pipeline.run()

        watcher = Watcher(pipeline=indexing_pipeline)
        observer = start_watcher_background(watcher)

        print("Welcome to the search engine! Type a query to search ('exit' or 'quit' to quit).\n")

        reranker_thread.join()
        search_pipeline = SearchPipeline(
            embedder=indexing_pipeline.embedder,
            reranker_model=reranker_model,
        )

        search_loop(search_pipeline)

    except KeyboardInterrupt:
        print("\nShutting down...")
    
    except Exception as e:
        print(f"Error: {e}")

    finally:
        if observer is not None:
            observer.stop()
            observer.join()


if __name__ == "__main__":
    main()