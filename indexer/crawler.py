# indexer/crawler.py

import os
import yaml


class Crawler:
    """
    Discovers files in configured directories and tracks which ones
    are new or modified using SHA-256 hashing.
    """

    def __init__(self, config_path="config.yaml"):
        """
        Load the config file and store the settings as instance variables.
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        self.watch_paths = config["watch_paths"]
        self.include_extensions = config["include_extensions"]
        self.skip_directories = config["skip_directories"]
        self.data_dir = config["data_dir"]

    def discover_files(self):
        """
        Walk through all watch_paths recursively and collect every file
        that matches include_extensions, skipping skip_directories.

        Returns:
            list[str] — list of absolute file paths
        """
        results=[]
        for path in self.watch_paths:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    if os.path.splitext(filename)[1] in self.include_extensions:
                        full_path = os.path.join(dirpath, filename)
                        results.append(full_path)
                dirnames[:] = [d for d in dirnames if d not in self.skip_directories]
        return results        

        
    def get_new_and_modified(self, known_file_info=None):
        """
        Compare discovered files against stored mtime+size to find
        which files are new or have been modified since last run.

        Args:
            known_file_info (dict) — {filepath: {"mtime": ..., "size": ...}}

        Returns:
            tuple: (files_to_process, current_file_info, deleted_files)
            - files_to_process: list[str] — paths that are new or changed
            - current_file_info: dict — {filepath: {"mtime": ..., "size": ...}}
            - deleted_files: set[str] — files that were deleted
        """
        if known_file_info is None:
            known_file_info = {}

        current_files = self.discover_files()
        files_to_process = []
        current_file_info = {}

        for filepath in current_files:
            stat = os.stat(filepath)
            mtime = stat.st_mtime
            size = stat.st_size
            current_file_info[filepath] = {"mtime": mtime, "size": size}

            known = known_file_info.get(filepath)
            if not known or known["mtime"] != mtime or known["size"] != size:
                files_to_process.append(filepath)

        deleted_files = set(known_file_info.keys()) - set(current_file_info.keys())

        return files_to_process, current_file_info, deleted_files


# --- Test ---
if __name__ == "__main__":
    crawler = Crawler()
    files = crawler.discover_files()
    print(f"Found {len(files)} files:")
    for f in files:
        print(f"  {f}")

    print("\n--- Checking for new/modified ---")
    to_process, hashes = crawler.get_new_and_modified()
    print(f"{len(to_process)} files to process")