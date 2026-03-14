# indexer/watcher.py

import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from indexer.pipeline import IndexingPipeline
import yaml


class IndexHandler(FileSystemEventHandler):
    """
    Handles filesystem events detected by watchdog.
    
    watchdog calls these methods automatically:
        - on_created(event)   → new file added
        - on_modified(event)  → existing file changed
        - on_deleted(event)   → file removed
    """

    def __init__(self, pipeline, config_path="config.yaml"):
        """
        Args:
            pipeline (IndexingPipeline) — existing pipeline instance
        """

        with open(config_path) as f:
            config = yaml.safe_load(f)
            self._debounce_seconds = config["debounce_seconds"]

        self.pipeline = pipeline
        self.include_extensions = self.pipeline.crawler.include_extensions
        self._last_event = {}    # {filepath: timestamp}

    def _is_duplicate(self, filepath):
        """
        Check if we've already handled an event for this file recently.
        Returns True if we should skip this event.
        """
        now = time.time()
        last = self._last_event.get(filepath, 0)
        if now - last < self._debounce_seconds:
            return True
        self._last_event[filepath] = now
        return False

    def _is_relevant(self, filepath):
        """
        Check if a file event is for a file type we care about.

        Args:
            filepath (str) — path from the event

        Returns:
            bool — True if the file extension is in our include list
        """
        ext = os.path.splitext(filepath)[1].lower()
        return ext in self.include_extensions

    def on_created(self, event):
        """
        Called when a new file is created.

        Args:
            event — watchdog event
        """
        if(event.is_directory):
            return
        
        if(not self._is_relevant(event.src_path)):
            return
        
        if self._is_duplicate(event.src_path):
            return

        print(f"New file detected: {event.src_path}")
        text = self.pipeline.extractor.extract(event.src_path)
        if(not text.strip()):
            print(f"  Skipping (no text extracted)")
            return
        
        chunks = self.pipeline.chunker.chunk_file(text, event.src_path)
        chunk_texts = [c["text"] for c in chunks]
        embeddings = self.pipeline.embedder.embed_chunks(chunk_texts)
        self.pipeline.store.remove_file_chunks(event.src_path)
        self.pipeline.store.add_chunks(chunks, embeddings)

        file_hash = self.pipeline.crawler.compute_hash(event.src_path)
        self.pipeline.store.save_file_info(event.src_path, file_hash, len(chunks))
        print(f"  File stored: {event.src_path}")


    def on_modified(self, event):
        """
        Called when an existing file is modified.

        Args:
            event - watchdog event
        """
        if(event.is_directory):
            return
        
        if(not self._is_relevant(event.src_path)):
            return
        
        if self._is_duplicate(event.src_path):
            return

        print(f"File modified: {event.src_path}")

        self.pipeline.store.remove_file_chunks(event.src_path)
        text = self.pipeline.extractor.extract(event.src_path)
        if(not text.strip()):
            print(f"  Skipping (no text extracted)")
            return
        
        chunks = self.pipeline.chunker.chunk_file(text, event.src_path)
        chunk_texts = [c["text"] for c in chunks]
        embeddings = self.pipeline.embedder.embed_chunks(chunk_texts)
        self.pipeline.store.add_chunks(chunks, embeddings)

        file_hash = self.pipeline.crawler.compute_hash(event.src_path)
        self.pipeline.store.save_file_info(event.src_path, file_hash, len(chunks))
        print(f"  File saved: {event.src_path}")

    def on_deleted(self, event):
        """
        Called when a file is deleted.

        Args:
            event - watchdog event
        """
        if(event.is_directory):
            return
        
        if(not self._is_relevant(event.src_path)):
            return

        print(f"File deleted: {event.src_path}")
        self.pipeline.store.remove_file_chunks(event.src_path)


class Watcher:
    """
    Starts watchdog observers on all configured watch_paths.
    Runs continuously until the user presses Ctrl+C.
    """

    def __init__(self, config_path="config.yaml"):
        """
        Initialize the Watcher.
        """
        self.pipeline = IndexingPipeline(config_path)
        self.handler = IndexHandler(self.pipeline)
        self.watch_paths = self.pipeline.crawler.watch_paths

    def start(self):
        """
        Start watching all configured directories.
        """
        observer = Observer()
        for path in self.watch_paths:
            observer.schedule(self.handler, path, recursive=True)
        observer.start()

        print(f"Watchdog active. Watching {', '.join(self.watch_paths)}")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping watcher...")
        finally:
            observer.stop()
            observer.join()


# --- Test it ---
if __name__ == "__main__":
    # First run the full pipeline to index existing files
    print("Running initial index...")
    watcher = Watcher()
    watcher.pipeline.run()

    # Then start watching for changes
    print("\nStarting file watcher...")
    watcher.start()