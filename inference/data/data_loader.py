import json
import csv
from pathlib import Path
import os
from typing import Dict, Any, Tuple




class CollectionLoader:
    def __init__(self, collection_path: str):
        self.collection = self.load_collection(collection_path)    

    def to_dict(self) -> Dict[str, Any]:
        return self.collection

    def load_collection(self, collection: Any) -> Dict[str, Any]:
        if isinstance(collection, str):
            if Path(collection).suffix == '.jsonl':
                return self._load_jsonl(collection)
            elif Path(collection).suffix == '.tsv':
                return self._load_tsv(collection)
            elif Path(collection).suffix == '.json':
                return self._load_json(collection)
            

    def _load_jsonl(self, collection_path: str) -> Dict[str, Any]:
        collection = {}
        with open(collection_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line.strip())
                collection[doc['_id']] = {
                    'document_id': doc['_id'],
                    'text': doc['text']
                }
        return collection

    def _load_tsv(self, collection_path: str) -> Dict[str, Any]:
        collection = {}
        with open(collection_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 2:
                    doc_id, text = row[0], row[1]
                    collection[doc_id] = {"id": doc_id, "text": text}
        return collection

    def _load_json(self, collection_path: str) -> Dict[str, Any]:
        with open(collection_path, 'r', encoding='utf-8') as f:
            collection = json.load(f)
        return collection

class QueryLoader:
    def __init__(self, queries_path: str):
        self.queries = self.load_queries(queries_path)


    def to_dict(self) -> Dict[str, Any]:
        return self.queries

    def load_queries(self, queries: Any) -> Dict[str, Any]:
        if isinstance(queries, str):
            if Path(queries).suffix == '.jsonl':
                return self._load_jsonl(queries)
            elif Path(queries).suffix == '.tsv':
                return self._load_tsv(queries)
            elif Path(queries).suffix == '.json':
                return self._load_json(queries)
            else:
                raise ValueError(f"Unsupported file extension: {Path(queries).suffix}")
        else:
            return queries
        

    def _load_jsonl(self, queries_path: str) -> Dict[str, Any]:
        queries = {}
        with open(queries_path, 'r', encoding='utf-8') as f:
            for line in f:
                query = json.loads(line.strip())
                queries[query['_id']] = {
                    '_id': query['_id'],
                    'question': query['text']
                }
        return queries

    def _load_tsv(self, queries_path: str) -> Dict[str, Any]:
        queries = {}
        with open(queries_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 2:
                    q_id, text = row[0], row[1]
                    queries[q_id] = {"_id": q_id, "question": text}
        return queries
    


class DataLoader:
    @classmethod
    def load_data(cls, collection_path: str, queries_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Main entry point for data loading"""
        # Get the absolute path to the project root
        project_root = Path(__file__).parent.parent.parent
        
        # Resolve paths relative to project root
        collection_path = os.path.join(project_root, collection_path)
        queries_path = os.path.join(project_root, queries_path)
        
        collection_ext = Path(collection_path).suffix
        queries_ext = Path(queries_path).suffix
        
        # Load raw data based on file extension
        collection_raw = cls._load_extension(collection_path, collection_ext)
        queries_raw = cls._load_extension(queries_path, queries_ext)
        
        # Process the raw data into standardized format
        collection = cls._process_collection(collection_raw)
        queries = cls._process_queries(queries_raw)
        
        return collection, queries

    @classmethod
    def _load_extension(cls, file_path: str, extension: str) -> Dict:
        """Load data based on file extension"""
        if extension == '.jsonl':
            return cls._load_jsonl(file_path)
        elif extension == '.tsv':
            return cls._load_tsv(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")

    @staticmethod
    def _load_jsonl(file_path: str) -> Dict:
        data = {}
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                data[item.get('id') or item.get('_id')] = item
        return data

    @staticmethod
    def _load_tsv(file_path: str) -> Dict:
        data = {}
        with open(file_path, 'r', newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 2:
                    doc_id, text = row[0], row[1]
                    data[doc_id] = {"id": doc_id, "text": text}
        return data

    @staticmethod
    def _process_collection(raw_data: Dict) -> Dict:
        """Process collection into standardized format"""
        processed = {}
        for doc_id, doc in raw_data.items():
            processed[doc_id] = {
                "id": doc_id,
                "text": doc.get("text") or doc.get("contents") or doc.get("body", ""),
                "title": doc.get("title", ""),
                # Add other fields as needed
            }
        return processed

    @staticmethod
    def _process_queries(raw_data: Dict) -> Dict:
        """Process queries into standardized format"""
        processed = {}
        for query_id, query in raw_data.items():
            processed[query_id] = {
                "query_id": query_id,
                "id": query_id,
                "text": query.get("text") or query.get("query") or query.get("question", ""),
            }
        return processed