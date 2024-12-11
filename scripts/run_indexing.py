# scripts/run_indexing.py
from models.indexer import ColBERTIndexer
from config import Config

if __name__ == "__main__":
    config = Config()
    indexer = ColBERTIndexer(config, config.COLLECTION_PATH)
    indexer.index_documents()