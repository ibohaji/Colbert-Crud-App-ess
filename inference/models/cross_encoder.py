from typing import List, Dict, Tuple
import torch
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder as SentenceCrossEncoder

# Import all necessary base classes and types
from .data_types import Query, Result, Document
from .reranker import Reranker
from .dense_retrieval import DenseRetrieval
from .bi_encoder import BiEncoder

class CrossEncoder(Reranker):
    def __init__(self, model_name: str, retrieval_model: DenseRetrieval, device: str):
        super().__init__(device)
        self.model_name = model_name
        self.retrieval_model = retrieval_model
        self.model = None  # Will be loaded later
        
    def create_index(self) -> None:
        """Create index using the underlying retrieval model"""
        self.retrieval_model.create_index()

    def search_all(self, queries:dict,  k: int = 1000) -> List[Tuple[str, str, int]]:
        all_results = []
        for _, query in queries.items():
            print(f"Searching for query: {query['_id']}")
            results = self.search(query, k)
            all_results.extend(results)

        return all_results

    def search(self, query: Query, k: int = 1000) -> List[Tuple[str, str, int]]:
        """First retrieve with base model, then rerank"""
        if not hasattr(self.retrieval_model, 'retrieve_top_k'):
            raise AttributeError("Retrieval model must implement retrieve_top_k method")
        
        query_text = query['text'] 
        
        candidate_doc = self.retrieval_model.retrieve_top_k(query_text, k)
        
        reranked_results = self.rerank(query, candidate_doc)
        return reranked_results

    def rerank(self, query: Query, candidates: List[Result]) -> List[Result]:
        """Rerank the candidates using cross-encoder"""

        pairs = [(query['text'], doc['text']) for doc in candidates]
        scores = []
        
        for query_text, doc_text in pairs:
            score = self.model.predict([query_text, doc_text])
            scores.append(score)
        
        reranked = []
        try:
            query_id = query["_id"]
        except KeyError:
            raise ValueError(f"Query ID is empty for query: {query}")

        if not query_id:
            raise ValueError("Query ID is empty")
        
        for score, doc in zip(scores, candidates):
            reranked.append({
                "_id": query_id,  
                "document_id": doc["document_id"],
                "score": float(score)
            })
        
        return self.rank(sorted(reranked, key=lambda x: x['score'], reverse=True))

    def load_model(self, device: str):
        """Load the cross-encoder model"""
        self.model = SentenceCrossEncoder(self.model_name, device=self.device)
        self.retrieval_model.load_model(self.device)

    def load_queries(self, path: str):
        self.queries = self.retrieval_model.load_queries(path)
        return self.queries
    
    def load_collection(self, path: str):
        self.collection = self.retrieval_model.load_collection(path)

    def get_index_size(self):
        """Get index size from retrieval model"""
        return self.retrieval_model.get_index_size()

    def get_model_dimension_embeddings(self):
        """Get embedding dimension from retrieval model"""
        if hasattr(self.retrieval_model, 'get_model_dimension_embeddings'):
            return self.retrieval_model.get_model_dimension_embeddings()
        return None