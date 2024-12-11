# models/updater.py
import argparse
from colbert import IndexUpdater
from models.searcher import ColBERTSearcher
from ..config import Config


class ColBERTUpdater:
    def __init__(self, config, searcher):
        self.config = config
        self.searcher = searcher

    def add_documents(self, new_docs):
        index_updater = IndexUpdater(config=self.config, searcher=self.searcher, checkpoint=self.config.CHECKPOINT)
        index_updater.add(new_docs)
        index_updater.persist_to_disk()
        print(f"Documents added to index: {new_docs}")

    def remove_documents(self, ids_to_remove: list):
        index_updater = IndexUpdater(config=self.config, searcher=self.searcher, checkpoint=self.config.CHECKPOINT)
        index_updater.remove(ids_to_remove)
        index_updater.persist_to_disk()
        print(f"Documents removed from index: {ids_to_remove}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Colbertv2-Update', description='ColBERT IR System')
    parser.add_argument('action', type=str, help='Action to perform (add/remove)')
    parser.add_argument('documents', type=str, help='Documents to add/remove')
    args = parser.parse_args()

    config = Config()
    searcher = ColBERTSearcher(config.INDEX_NAME)
    updater = ColBERTUpdater(config, searcher)

    if args.action == 'a':
        updater.add_documents(args.document)

    elif args.action == 'r':
        updater.remove_documents(args.document)

