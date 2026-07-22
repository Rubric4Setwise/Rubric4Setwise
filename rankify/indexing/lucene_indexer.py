#lucene_indexer.py
import shutil
import subprocess
import logging

from pathlib import Path

from rankify.indexing.base_indexer import BaseIndexer
from rankify.indexing.format_converters import to_pyserini_jsonl

logging.basicConfig(level=logging.INFO)

class LuceneIndexer(BaseIndexer):
    """
    Lucene Indexer for creating and loading Lucene indices using Pyserini.
    This class handles the preparation of the corpus, building the index, and loading the index.
    """
    def __init__(self, corpus_path, output_dir="rankify_indices", chunk_size=1024, threads=32, index_type="wiki",
                 retriever_name="bm25",**kwags):
        super().__init__(corpus_path, output_dir, chunk_size, threads, index_type, retriever_name)
        self.id_mapping = {}  # string_id -> integer_id mapping
        self.mapping_file = self.output_dir / "id_mapping.json"
        
        if index_type =="wiki" and retriever_name == "bm25":
            self.index_dir = self.output_dir / f"bm25_index"
        else:
            self.index_dir = self.output_dir / f"{retriever_name}_index_{index_type}"
    
    def _create_id_mapping(self):
        """
        Pre-scan the corpus to create string ID -> integer ID mapping.
        """
        import json
        
        logging.info("Creating ID mapping...")
        self.id_mapping = {}
        next_id = 0
        
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    doc = json.loads(line.strip())
                    original_id = doc.get("id")
                    if original_id and original_id not in self.id_mapping:
                        self.id_mapping[str(original_id)] = next_id
                        next_id += 1
                except:
                    continue
    
        logging.info(f"Created mapping for {len(self.id_mapping)} unique IDs")
    def _save_id_mapping(self):
        """Save the ID mapping to a JSON file."""
        import json
        with open(self.mapping_file, "w", encoding="utf-8") as f:
            json.dump(self.id_mapping, f, ensure_ascii=False, indent=2)
        logging.info(f"ID mapping saved to {self.mapping_file}")
    def _save_pyserini_corpus(self):
        """
        Convert the corpus to the Pyserini JSONL format.
        This method uses the `to_pyserini_jsonl` function to convert the corpus file
        :return: str - The path to the converted Pyserini JSONL corpus file.
        """
        return to_pyserini_jsonl(self.corpus_path, self.output_dir, self.chunk_size, self.threads)
       
    def build_index(self):
        """
        Build the Lucene index from the corpus.
        """
        # 1) Create & save mapping BEFORE converting corpus (so converter can load it)
        self._create_id_mapping()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._save_id_mapping()

        # 2) Convert corpus (may return str or Path depending on your converter)
        corpus_path = Path(self._save_pyserini_corpus())

        # 3) Prep temp dir
        temp_corpus_dir = self.output_dir / "temp_corpus"
        if temp_corpus_dir.exists():
            shutil.rmtree(temp_corpus_dir)
        temp_corpus_dir.mkdir(parents=True)

        corpus_file_path = temp_corpus_dir / "corpus.json"

        # 4) Move the file into the temp dir (works across filesystems)
        shutil.move(str(corpus_path), str(corpus_file_path))

        # 5) Reset index dir
        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
        self.index_dir.mkdir(parents=True)

        # 6) Run Pyserini
        cmd = [
            "python", "-m", "pyserini.index.lucene",
            "-collection", "JsonCollection",
            "-generator", "DefaultLuceneDocumentGenerator",
            "-threads", str(self.threads),
            "-input", str(temp_corpus_dir),
            "-index", str(self.index_dir),
            "-storePositions", "-storeDocvectors", "-storeRaw"
        ]
        logging.info(f"Running Pyserini indexer:\n{' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        # 7) Cleanup & finish
        shutil.rmtree(temp_corpus_dir)
        self._save_title_map()
        logging.info(f"Indexing complete. Index stored at {self.index_dir}")

    def load_index(self):
        """
        Load the Lucene index from the specified directory.

        This method checks if the index directory exists and is not empty,
        and raises an error if the index is locked or not found.
        """
        if not self.index_dir.exists() or not any(self.index_dir.iterdir()):
            raise FileNotFoundError(f"Index directory {self.index_dir} does not exist or is empty.")
        logging.info(f"Index loaded from {self.index_dir}")

