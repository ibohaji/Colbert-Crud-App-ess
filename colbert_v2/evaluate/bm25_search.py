# bm25_search.py 
import argparse 
from elasticsearch import Elasticsearch, helpers
from colbert_v2.train.distillation.el_search import EsSearcher
from colbert_v2.custom.data_organizer import GenQueryData, CollectionData
import csv 
import time 


def search_bm25_results(collection, output_path, queries):
    queries = GenQueryData(queries) 
    collection = CollectionData(collection)

    with open(output_path, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')

        with EsSearcher() as es_searcher:

            es_searcher.index_documents(collection.collection_dict)

            start = time.time()

            for query_id, query in queries.queries_dict.items():
        
                docs = es_searcher.retrieve(index='documents', query_text=query["text"], positive_doc_id=999999999, num_results=1000)

                for rank, hit in enumerate(docs, start=1):
                    doc_id = hit['_id']
                    writer.writerow([query_id, doc_id, rank])

        
        end = time.time() 
        total = end - start 

    print(f"total search time: {total}")
    print("Successfully indexed and retrieved files")

            


if __name__=="__main__":

    parser = argparse.ArgumentParser(description = "Retrieve data using bm25")

    parser.add_argument(
        '--collection'
    )
    parser.add_argument(
        '--output_path'
    )
    parser.add_argument(
        '--queries'
    )

    parser.add_argument(
        '--data_name'
    )

    args = parser.parse_args()
    search_bm25_results(args.collection, args.output_path, args.queries ) 

