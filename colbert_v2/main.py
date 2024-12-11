# main.py
from colbert_v2.models.indexer import ColBERTIndexer
from config import Config
from models.searcher import ColBERTSearcher
from models.updater import ColBERTUpdater

if __name__ == "__main__":
    print("Welcome to ColBERT IR System")
    config = Config()
    searcher = ColBERTSearcher(config.INDEX_NAME)

    while True:
        print("\n1. Index Documents\n2. Search\n3. Update Index\n4. Exit")
        choice = int(input("Select an option: "))

        if choice == 1:
            indexer = ColBERTIndexer(config, config.COLLECTION_PATH)
            indexer.index_documents()

        elif choice == 2:
            query = input("Enter search query: ")
            searcher.search(query)

        elif choice == 3:
            updater = ColBERTUpdater(config, searcher)
            action = input("Add or Remove documents? (a/r): ")

            if action == 'a':
                new_docs = input("Enter new documents (comma-separated): ").split(',')
                updater.add_documents(new_docs)
            elif action == 'r':
                remove_ids = list(map(int, input("Enter IDs to remove (comma-separated): ").split(',')))
                updater.remove_documents(remove_ids)

        elif choice == 4:
            print("Exiting...")
            break
