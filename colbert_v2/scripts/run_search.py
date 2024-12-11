# scripts/run_search.py
from config import Config
from models.searcher import ColBERTSearcher

if __name__ == "__main__":
    query = input("Enter search query: ")
    searcher = ColBERTSearcher(Config.INDEX_NAME)
    searcher.search(query)
