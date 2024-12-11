from sentence_transformers import SentenceTransformer, util
import torch
from typing import List, Dict, TypedDict, Tuple
import json

# Import base classes and types
from .data_types import Result, Document, Query
from .retriever import Retriever
from .dense_retrieval import DenseRetrieval

class BiEncoder(Retriever):
    def __init__(self, model_name, device):

        self.model_name = model_name
        self.device = device

    def _search(self, query: Query, k: int = 1000) -> List[Result]:
        """Internal search method that returns full results"""
        return self.retrieve_top_k(query['text'], k)

    def search(self, query: Query, k: int = 10) -> List[Tuple[str, str, int]]:
        """Public search method that returns ranked results"""
        results = self._search(query, k)
        return self.rank(results)
    
    def retrieve_top_k(self, query_text: str, k: int = 1000) -> List[Result]:
        query_embedding = self.model.encode(
            query_text, 
            convert_to_tensor=True,
            device=self.device
        )
        scores = util.cos_sim(query_embedding, self.embeddings)[0]
        scores = scores.cpu()
        k = min(k, len(self.embeddings))
        top_k_idx = torch.topk(scores, k).indices
        
        return [
            {
                "document_id": self.documents[idx]["document_id"],
                "text": self.documents[idx]["text"],
                "score": float(scores[idx].item())
            }
            for idx in top_k_idx
        ]

    def create_index(self) -> None:
        batch_size = 132
        documents = list(self.collection.values())
        
        try:
            self.embeddings = self.model.encode(
                [doc["text"] for doc in documents],
                convert_to_tensor=True,
                device=self.device,
                batch_size=batch_size,
                show_progress_bar=True
            )
            self.documents = documents
        except RuntimeError as e:  # If we run out of memory
            print("GPU memory exceeded, falling back to smaller batches")

    def load_queries(self, Queries) -> Dict[str, Query]:
        if isinstance(Queries, str):
            queries = {}
            with open(Queries, 'r', encoding='utf-8') as f:
                for line in f:
                    query = json.loads(line.strip())
                    queries[query['_id']] = {
                        '_id': query['_id'],
                            'text': query['text']
                            }
            self.queries = queries
        else:
            self.queries = Queries

        return self.queries
    
    def load_collection(self, path: str) -> Dict[str, Document]:
        collection = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line.strip())
                collection[doc['_id']] = {
                    'document_id': doc['_id'],
                    'text': doc['text']
                }
        self.collection = collection
        return collection


    def get_index_size(self):
        """
        returns the size of the embeddings in mb
        """
       
        bytes_size = self.embeddings.element_size() * self.embeddings.nelement()
        mb_size = bytes_size / (1024 * 1024)  # Convert bytes to MB
        
        return mb_size
    
    def load_model(self, device: str):
        self.model = SentenceTransformer(self.model_name, device=device)    


    def get_model_dimension_embeddings(self):
        return self.model.get_sentence_embedding_dimension()



