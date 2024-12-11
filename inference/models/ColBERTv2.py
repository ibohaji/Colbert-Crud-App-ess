from colbert_v2.models.indexer import ColBERTIndexer
from colbert_v2.models.searcher import ColBERTSearcher
from colbert.infra import ColBERTConfig, Run, RunConfig
from time import time
import os
import csv
from typing import Dict, Any, Tuple, List
from pathlib import Path
import json
from colbert import Searcher
from colbert import Indexer
from typing import Tuple
from .data_types import Query, Document, Result
from .dense_retrieval import DenseRetrieval
from ..utils.collection_utils import standardize_collection_ids, map_back_results
import os
import subprocess
import sys
import torch



class ColBERTv2(DenseRetrieval):
    def __init__(self, config, collection_path, device):
        super().__init__()
        self.device = device
        self.collection_path = collection_path
        self.checkpoint = config.CHECKPOINT
        self.config = config
        self.queries = {}  
        self.indexed = False 
        
        # Set CUDA visibility based on device choice

        if device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["COLBERT_LOAD_TORCH_EXTENSION_VERBOSE"] = "True"
            # Disable JIT/CUDA compilation attempts
            os.environ["TORCH_CUDA_ARCH_LIST"] = ""
            os.environ["FORCE_CPU"] = "True"

        
        # RunConfig only needs experiment info
        self.run_config = RunConfig(
            nranks=1, 
            experiment='experiments')
        
        
    def load_collection(self, path: str) -> Dict:
        collection = super().load_collection(path)
        # Standardize IDs and keep both mappings
        self.collection, self.id_mapping, self.reverse_mapping = standardize_collection_ids(collection)
        return self.collection

    def load_queries(self, queries: str) -> Dict:
        """Load and standardize queries"""
        if isinstance(queries, str):
            with open(queries, 'r', encoding='utf-8') as f:
                queries = {}
                for line in f:
                    query = json.loads(line.strip())
                    query_id = query.get('_id') or query.get('query_id')
                    if query_id and 'text' in query:
                        queries[query_id] = {
                            'query_id': query_id,
                            'question': query['text']
                        }
                
            self.queries = queries
            return queries

    def _convert_to_tsv(self, collection: Dict, collection_path: str) -> Path:
        """Convert collection dictionary to TSV format that ColBERT expects"""
        tsv_path = Path(collection_path).parent / "collection.tsv"
        
        with open(tsv_path, 'w', encoding='utf-8') as tsv_out:
            for doc_id, doc in collection.items():
                tsv_out.write(f"{doc_id}\t{doc['text']}\n")
        
        return tsv_path

    def create_index(self) -> None:
        """Create index from the collection"""
    
        tsv_path = self._convert_to_tsv(self.collection, self.collection_path)
        print(f"the path is: {tsv_path}")
        self.load_model(self.device)
        with Run().context(self.run_config):
            config = ColBERTConfig(
            root='experiments', 
                nbits=2        )
            self.indexer = Indexer(checkpoint=self.checkpoint, config=config)

            self.indexer.index(
                name=self.config.INDEX_NAME,
                collection=str(tsv_path),
                overwrite=True
            )

    
    def search_all(self, queries, k: int = 1000) -> List[Result]:
        with Run().context(self.run_config):
            config = ColBERTConfig(
                root='experiments'
            )
            
            self.searcher = Searcher(
                index=self.config.INDEX_NAME, 
                config=config, 
                checkpoint=self.checkpoint,
                use_gpu=(self.device != "cpu")  # Explicitly set GPU usage
            )
            results = self.searcher.search_all(queries, k)
        
            return map_back_results(results, self.reverse_mapping)

    def search(self, query: Query, k: int = 1000) -> List[Result]:
        if not self.queries:
            raise ValueError("Queries not loaded. Call load_queries first.")
        
        query_text = query['question']
        with Run().context(RunConfig(nranks=1, experiment='experiments')):
                      results = self.searcher.search(query_text, k)

        # Map back to original IDs before returning
        return map_back_results(results, self.reverse_mapping)


    def load_indexer(self):
        config = ColBERTConfig(root='experiments')
        self.indexer = Indexer(checkpoint=self.checkpoint, config=config)

    def load_model(self, device):
        return 

    def evaluate(self, queries, qrel):
        pass

    def memory_usage(self):
        custom_memory = super().memory_usage()
        custom_memory["gpu"] = "Track GPU-specific memory here"
        return custom_memory

    def get_model_dimension_embeddings(self):
        return 128 # Todo: Check this
    def get_index_size(self):
        index_path = os.path.join('experiments','experiments','indexes','crud.colbert.index.inference')
        
        total_size = 0
        
        for dirpath, dirnames, filenames in os.walk(index_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                total_size += os.path.getsize(file_path)

        return total_size / (1024 * 1024 * 1024)  # Convert bytes to GB

    def process_data(self, collection: Dict, queries: Dict) -> Tuple[Dict, Dict]:
        """Convert standardized format to ColBERT format"""
        colbert_collection = {
            doc_id: {
                "_id": doc_id,
                "text": doc["text"]
            }
            for doc_id, doc in collection.items()
        }
        
        colbert_queries = {
            qid: {
                "qid": qid,
                "question": query["text"]
            }
            for qid, query in queries.items()
        }
        
        return colbert_collection, colbert_queries

