import logging

import tqdm

from .el_search import EsSearcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HardNegativesSampler:
    def __init__(self, queries, collection, es):
        self.collection = collection
        self.es = es
        self.index_name = 'documents'
        self.queries = queries
        self.setup_es_index()

    def setup_es_index(self):
        try:
            self.es.index_documents(self.collection.collection_dict)
        except Exception as e:
            logger.error(f"Error indexing documents into Elasticsearch: {e}")
            raise e
        logger.info(f"Indexed {len(self.collection.collection_dict)} documents into Elasticsearch")

    def get_hard_negatives_all(self, num_negatives=3):
        hard_negatives = {}
        for query_id, item in tqdm.tqdm(self.queries.queries_dict.items()):
            query_text = item['text']
            positive_doc_id = item['doc_id']
            negatives = self.get_hard_negative_pids(query_text, positive_doc_id, num_negatives)
            hard_negatives[query_id] = negatives
        return hard_negatives

    def get_hard_negative_pids(self, query_text, positive_doc_id, num_negatives):
        try:
            hits = self.es.retrieve(index=self.index_name, query_text=query_text, positive_doc_id=positive_doc_id, num_results=num_negatives)
            hit_ids = [hit['_id'] for hit in hits]
            logger.info(f"Search Response for Query '{query_text}': {hit_ids}")
            return hit_ids
        except Exception as e:
            logger.error(f"Error searching for hard negatives: {e}")
            return []
