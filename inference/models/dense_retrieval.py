from abc import ABC, abstractmethod
import psutil
import json
from typing import Dict, List, Tuple
from pathlib import Path
from datetime import datetime
from .data_types import Query, Document, Result

class DenseRetrieval(ABC):
    def load_queries(self, Queries) -> Dict[str, Query]:
        if isinstance(Queries, str):
            queries = {}
            with open(Queries, 'r', encoding='utf-8') as f:
                for line in f:
                    query = json.loads(line.strip())
                    queries[query['_id']] = {
                    '_id': query['_id'],
                    'question': query['text']
            }
            self.queries = queries
            print(f"Loaded {len(queries)} queries for {self.__class__.__name__}")
        else:
            self.queries = Queries

        return self.queries
        
    
    def load_collection_jsonl(self, Collection):
        
            collection = {}
            with open(Collection, 'r', encoding='utf-8') as f:
                for line in f:
                    doc = json.loads(line.strip())
                    collection[doc['_id']] = {
                    'document_id': doc['_id'],
                    'text': doc['text']
                }
                
            self.collection = collection
            return self.collection


    def load_collection_json(self, Collection): 
        collection = {}
        with open(Collection, "r", encoding="utf-8") as f: 
            col = json.load(f)
            for key, value in col.items():
                if not isinstance(value, dict):
                    print(f"Skipping non-dict entry for key {key}: {value}")
                    continue

                title = " ".join(str(value.get("title", "")).split())
                text = " ".join(str(value.get("text", "")).split())

                # Combine title and text, then ensure it is stripped
                cleaned_text = f"{title} {text}".strip()

                # Add to collection if there's any valid content
                if cleaned_text:
                    collection[key] = { 
                        "document_id": key,
                        "text": cleaned_text
                    }
                else:
                    print(f"Skipping entry with no content for key {key}: {value}")

        self.collection = collection
        return self.collection


    def load_collection(self, Collection) -> Dict[str, Document]:

        if isinstance(Collection, str):
            suff = Path(Collection).suffix  
            print(f"suffix is: {suff}")

            if suff is not None:
                if suff == ".jsonl":
                    return self.load_collection_jsonl(Collection)
                if suff ==".json":
                    return self.load_collection_json(Collection)
            
        else:
            self.collection = Collection

        self.collection = Collection 

        return self.collection

    
    def load_collection_tsv(Collection):
        collection = {} 

        with open(Collection, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                pid, text = row[0], row[1]

    

    @abstractmethod
    def create_index(self, dataset):
        pass

    @abstractmethod
    def search(self, query: Query, k: int = 10) -> List[Tuple[str, str, int]]:
        pass

    @abstractmethod
    def load_model(self, device):
        pass

    @abstractmethod
    def get_index_size(self):
        pass

    @abstractmethod
    def get_model_dimension_embeddings(self):
        pass

    def rank(self, results: List[Result]) -> List[Tuple[str, str, int]]:
        """Convert results to ranking format (qid, pid, rank)"""
        rankings = []
        for rank, result in enumerate(results, start=1):
            rankings.append((
                result['query_id'],
                result['document_id'],
                rank
            ))
        return rankings

    def save_results(self, results: List[Tuple[str, str, int]]):
        """Save rankings to inference/measured_data/model_name/timestamp/results.tsv"""
        model_name = self.__class__.__name__.lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get path relative to inference directory
        current_file = Path(__file__)
        inference_dir = current_file.parent.parent
        base_dir = inference_dir / "measured_data" / model_name / timestamp
        base_dir.mkdir(parents=True, exist_ok=True)
        
        rankings_path = base_dir / "results.tsv"
        with open(rankings_path, 'w', encoding='utf-8') as f:
            for qid, pid, rank in results:
                f.write(f"{qid}\t{pid}\t{rank}\n")
        
        return timestamp

    def __repr__(self):
        return self.__class__.__name__ 