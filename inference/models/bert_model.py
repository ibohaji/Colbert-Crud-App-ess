from abc import ABC, abstractmethod
import psutil
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from collections import defaultdict
import sqlite3
import os
from tqdm import tqdm
from typing import List, Dict, TypedDict, Tuple
import json
try:
    import faiss
except ImportError:
    raise ImportError("Please install FAISS: pip install faiss-cpu")

from .data_types import Query, Document, Result
from .dense_retrieval import DenseRetrieval
from .reranker import Reranker

class BERTRetriever(DenseRetrieval):
    def __init__(self, model_name, device):
        self.device = device
        self.checkpoint = model_name
        self.gpu_index = device == "cuda"
        self.setup_faiss(device)
        
    def setup_faiss(self, device):
        """Initialize FAISS for CPU"""
        self.gpu_index = False  # Always use CPU index
        self.index = None

    def search_all(self, queries:dict, k: int = 1000) -> List[Tuple[str, str, int]]:
        all_results = []
        for query_id, query_text in queries.items():
            results = self.search(query_text, k)
            all_results.extend(results)

        return all_results

    def _search(self, query: Query, k: int = 10) -> List[Result]:
        """Internal search method that returns full results"""
        
        query_text = query.get('text', '')
        query_id = query.get('_id', '')
        
        if not query_text:
            raise ValueError("Query text is empty")
        if not query_id:
            raise ValueError("Query ID is empty")
        
        query_embedding = self.model.encode(
            query_text, 
            convert_to_tensor=True,
            show_progress_bar=False,
            device=self.device
        ).cpu().numpy().astype('float32')
        
        query_embedding = query_embedding.reshape(1, -1)
        
        if self.gpu_index:
            query_embedding = faiss.vector_to_array(query_embedding)
            
        scores, doc_ids = self.index.search(query_embedding, k)
        
        results = []
        for doc_id, score in zip(doc_ids[0], scores[0]):
            doc_id_str = str(doc_id)
            if doc_id_str in self.collection:
                results.append({
                    "query_id": query_id,
                    "document_id": doc_id_str,
                    "score": float(score)
                })
        
        return results

    def search(self, query: Query, k: int = 10) -> List[Tuple[str, str, int]]:
        """Public search method that returns ranked results"""
        results = self._search(query, k)
        return self.rank(results)

    def rank(self, results: List[Result]) -> List[Tuple[str, str, int]]:
        """Convert results to ranked format (qid, pid, rank)"""
        if not results:
            return []
        
        # Sort by score in descending order
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        # Convert to ranking format (query_id, document_id, rank)
        rankings = [
            (doc['query_id'], doc['document_id'], rank + 1)
            for rank, doc in enumerate(sorted_results)
        ]
        
        return rankings

    def get_index_size(self):
        """Returns the approximate size of the FAISS index in gigabytes"""
        # Get number of vectors and dimension
        n_vectors = self.index.ntotal
        dimension = self.index.d
        
        # Calculate size in bytes (4 bytes per float32)
        size_bytes = n_vectors * dimension * 4
        
        # Convert to megabytes
        size_mb = size_bytes / (1024 ** 2)
        
        return size_mb

    def get_model_dimension_embeddings(self):
        """Returns the dimension of embeddings produced by the model"""
        return self.model.get_sentence_embedding_dimension()

    def create_index(self):
        """Create index from the stored collection"""
        documents = []
        for doc in self.collection.values():
            if isinstance(doc, dict) and 'text' in doc:
                documents.append(doc['text'])
            elif isinstance(doc, str):
                documents.append(doc)
            else:
                raise ValueError(f"Invalid document format: {doc}")
        
        embeddings = self.model.encode(documents, 
                                     convert_to_tensor=True,
                                     show_progress_bar=True,
                                     device=self.device)
        
        # Move to CPU before converting to numpy
        embeddings = embeddings.cpu().numpy().astype('float32')
        
        # Dynamically get embedding dimension from the first embedding
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        
        if self.gpu_index:
            self.index = faiss.index_cpu_to_gpu(self.res, 0, index)
        else:
            self.index = index
            
        self.index.add(embeddings)
        return self.index


    def load_model(self, device):
        self.model = SentenceTransformer("sentence-transformers/msmarco-bert-base-dot-v5")
        self.model = self.model.to(self.device)
        
    def evaluate(self, queries, qrel):
        pass

    def memory_usage(self):
        custom_memory = super().memory_usage()
        custom_memory["gpu"] = "Track GPU-specific memory here" # TODO
        return custom_memory


    def get_max_seq_length(self):
        """Returns the maximum sequence length the model can handle"""
        return self.model.max_seq_length

    def retrieve_top_k(self, query_text: str, k: int = 10) -> List[Result]:
        """Retrieve top k documents for a query text"""
        if not query_text:
            raise ValueError("Query text is empty")
        
        # Encode query
        query_embedding = self.model.encode(
            query_text, 
            convert_to_tensor=True,
            show_progress_bar=False,
            device=self.device
        ).cpu().numpy().astype('float32')
        
        query_embedding = query_embedding.reshape(1, -1)
        
        if self.gpu_index:
            query_embedding = faiss.vector_to_array(query_embedding)
            
        # Search index
        scores, doc_ids = self.index.search(query_embedding, k)
        
        # Convert to results format
        results = []
        for doc_id, score in zip(doc_ids[0], scores[0]):
            doc_id_str = str(doc_id)
            if doc_id_str in self.collection:
                results.append({
                    "document_id": doc_id_str,
                    "text": self.collection[doc_id_str]['text'],
                    "score": float(score)
                })
        
        return results


class BERTReranker(Reranker):
    def __init__(self, model_name, device):
        super().__init__(device)
        self.checkpoint = model_name
        
    def rerank(self, query: Query, candidates: List[Result]) -> List[Result]:
        pairs = [(query['text'], doc['text']) for doc in candidates]
        scores = self.model.encode(pairs)
        
        reranked = []
        for score, doc in zip(scores, candidates):
            reranked.append({
                **doc,
                "score": float(score)
            })
        
        return sorted(reranked, key=lambda x: x['score'], reverse=True)

    def load_model(self, device):
        self.model = SentenceTransformer(self.checkpoint)
        self.model = self.model.to(device)

