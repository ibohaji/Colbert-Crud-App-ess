# scripts/run_search.py
from models.searcher import ColBERTSearcher
from config import Config

if __name__ == "__main__":
    query = input("Enter search query: ")
    searcher = ColBERTSearcher(Config.INDEX_NAME)
    searcher.search(query)
