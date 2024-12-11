from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Union, Any
import numpy as np
import json
from pathlib import Path
import random
from .dense_retrieval import DenseRetrieval
from .data_types import Query, Document, Result
from datetime import datetime
import os
import time
from collections import defaultdict
import math
import string

class LexicalRetrieval(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        pass

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def get_tokenizer(self):
        pass
    
    def save_results(self, results):
        model_name = self.__class__.__name__.lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        base_dir = Path("measured_data") / model_name / timestamp
        base_dir.mkdir(parents=True, exist_ok=True)
        
        rankings_path = base_dir / "results.tsv"
        with open(rankings_path, 'w', encoding='utf-8') as f:
            for qid, result in results.items():
                for pid, score in result.items():
                    f.write(f"{qid}\t{pid}\t{score}\n")
        
        return timestamp

class BM25Retriever(LexicalRetrieval):
    def __init__(self):
        self.collection = None
        self.queries = None
        self.inverted_index = defaultdict(dict)
        self.doc_lengths = {}
        self.avg_doc_length = 0.0
        self.N = 0
        self.doc_map = {}
        self.k1 = 1.2
        self.b = 0.75
        self.idf = {}
        self.epsilon = 1e-6
        self.tokenized_docs = []
        self.doc_lookup = {}
        self.bm25 = None
        self.indexed = False


    def load_collection_depricated(self, Collection) -> Dict[str, Document]:
        """Load collection and build index"""
        if isinstance(Collection, str):
            collection = {}
            with open(Collection, 'r') as f:
                for line in f:
                    doc = json.loads(line.strip())
                    collection[doc['_id']] = {
                        'document_id': doc['_id'],
                        'text': doc["title"] + " " + doc['text'] 
                    }
            self.collection = collection
        else:
            self.collection = Collection

        return self.collection

    def retrieve(self, query: str, k: int = 1000) -> List[Tuple[str, float]]:
        """Retrieves documents matching the query using BM25 scoring."""
        query_terms = self.preprocess_text(query)
        scores = defaultdict(float)

        # Calculate BM25 scores for each document
        for term in query_terms:
            if term in self.inverted_index:
                idf = self.idf.get(term, 0)
                for doc_id, freq in self.inverted_index[term].items():
                    doc_len = self.doc_lengths[doc_id]
                    score = self._calculate_bm25_score(freq, doc_len, idf)
                    scores[doc_id] += score

    # Sort by score and return top k results
        ranked_docs = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]
        return [
            (doc_id, float(score))
            for doc_id, score in ranked_docs
        ]





    def _calculate_idf(self):
        """Calculates inverse document frequency (IDF) for all terms."""
        for term, doc_dict in self.inverted_index.items():
            df = len(doc_dict)  # Number of documents containing the term
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            self.idf[term] = idf

    def preprocess_text(self, text):
        """Preprocesses text by lowercasing and removing punctuation."""
        translator = str.maketrans("", "", string.punctuation)
        return text.lower().translate(translator).split()

    def _calculate_bm25_score(self, tf, doc_len, idf):
        """Calculates BM25 score for a single term-document pair."""
        # Normalize term frequency by document length
        norm_len = doc_len / self.avg_doc_length
        numerator = tf * (self.k1 + 1)
        denominator = tf + self.k1 * (1 - self.b + self.b * norm_len)

        if abs(denominator) < self.epsilon:
            denominator = self.epsilon

        return idf * (numerator / denominator)


    def _search(self, query: Query, k: int = 1000) -> List[Result]:
        """
        Internal search method using Lucene-style BM25 scoring
        """
        query_text = query['question']
        query_id = query['_id']

        # Preprocess query terms
        query_terms = self.preprocess_text(query_text)
        scores = defaultdict(float)
        
        for term in query_terms:
            if term in self.inverted_index:
                idf = self.idf.get(term, 0)
                
                # Skip terms with 0 IDF (appear in all docs)
                if idf == 0:
                    continue
                    
                for doc_id, freq in self.inverted_index[term].items():
                    doc_len = self.doc_lengths[doc_id]
                    score = self._calculate_bm25_score(freq, doc_len, idf)
                    scores[doc_id] += score

        # Sort by score and limit to top k
        ranked_docs = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]

        return [
            {
                "document_id": doc_id,
                "score": float(score)
            }
            for doc_id, score in ranked_docs
        ]
        
    def rank(self, results: List[Dict[str, float]]) -> Dict[str, int]:
   
        sorted_results = sorted(results, key=lambda x: -x['score'])
        ranked_results = {
            result["document_id"]: rank
            for rank, result in enumerate(sorted_results, start=1)
        }
        return ranked_results

    def create_index(self):
        """Index documents by building an inverted index and calculating IDF."""

        if not(self.indexed):
                
            documents = self.collection.values()
            total_doc_length = 0

            for doc in documents:
                doc_id = doc['document_id']
                text = doc['text']
                self.doc_lookup[doc_id] = {"document_id": doc_id, "text": text}

                # Preprocess and tokenize
                tokens = self.preprocess_text(text)
                self.doc_lengths[doc_id] = len(tokens)
                total_doc_length += len(tokens)
                term_freqs = defaultdict(int)

                # Count term frequencies in this document
                for term in tokens:
                    term_freqs[term] += 1

                # Update the inverted index
                for term, freq in term_freqs.items():
                    self.inverted_index[term][doc_id] = freq

            self.N = len(self.doc_lengths)
            self.avg_doc_length = total_doc_length / self.N if self.N > 0 else 0

            # Precompute IDF for all terms
            self._calculate_idf()
            self.indexed = True

    def search(self, query: Query, k: int = 1000) -> List[Dict[str, Union[str, float]]]:
        results = self._search(query, k)
        formatted_results =  {
            result["document_id"]: result["score"]  
                for result in results 
                              }
    

        #ranked_results = self.rank(results)
        return formatted_results

    def search_all(self, queries: dict, k: int = 1000) -> Dict[str, List[Result]]:
        all_results = {query_id: self.search(query, k) for query_id, query in queries.items()}

        return all_results
    

    def retrieve_top_k(self, query_text: str, k: int = 1000) -> List[Result]:
        if not self.bm25:
            raise ValueError("Index not created. Call create_index() first.")

        tokenized_query = self.tokenize(query_text)
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[-k:][::-1]
        
        return [
            {
                "document_id": self.doc_lookup[idx]["document_id"],
                "text": self.doc_lookup[idx]["text"],
                "score": float(scores[idx])
            }
            for idx in top_indices
        ]

    def add_documents(self, documents: List[Document]) -> None:
        if self.bm25 is None:
            self.create_index(documents)
            return

        new_tokenized = [self.tokenize(doc['text']) for doc in documents]
        
        start_idx = len(self.doc_lookup)
        for i, doc in enumerate(documents):
            self.doc_lookup[start_idx + i] = doc
        
        self.tokenized_docs.extend(new_tokenized)
        
        self.bm25 = BM25Okapi(self.tokenized_docs)

    
    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        
        text = ''.join(c for c in text if c.isalnum() or c.isspace())
        
        return text.split()
    
    def get_index_size(self) -> float:
        """Calculate approximate size of BM25 index in MB"""
        if not self.tokenized_docs:
            return 0.0
        
        # Calculate size of tokenized documents
        token_size = sum(len(' '.join(doc).encode('utf-8')) for doc in self.tokenized_docs)
        
        # Calculate size of document lookup
        lookup_size = sum(len(str(doc).encode('utf-8')) for doc in self.doc_lookup.values())
        
        # Calculate size of BM25 parameters (approximate)
        if self.bm25:
            bm25_params_size = (
                len(self.tokenized_docs) * 4 +  # document lengths
                len(set(word for doc in self.tokenized_docs for word in doc)) * 8  # idf scores
            )
        else:
            bm25_params_size = 0
            
        total_bytes = token_size + lookup_size + bm25_params_size
        return total_bytes / (1024 * 1024)  # Convert to MB

    def get_model_dimension_embeddings(self) -> int:
        """BM25 doesn't use embeddings, return vocabulary size instead"""
        if self.tokenized_docs:
            return len(set(word for doc in self.tokenized_docs for word in doc))
        return 0
    
    def load_model(self, device):
        pass 
    
    def load_queries(self, Queries: Any) -> Dict[str, Query]:
        """Load queries from file - simple text-based loading"""
        if isinstance(Queries, str):
            queries = {}
            with open(Queries, 'r') as f:
                for line in f:
                    query = json.loads(line.strip())
                    queries[query['_id']] = {
                        'query_id': query['_id'],
                        'question': query['text']
                        } 
                
            self.queries = queries
        else:
            self.queries = Queries

        return self.queries
    
    def load_collection(self, Collection) -> Dict[str, Document]:
        """Load collection from file"""
        if isinstance(Collection, str):

            collection_ = {}
            with open(Collection, 'r') as f:
                doc = json.load(f)
                
                for doc_id, values in doc.items():

                    collection_[doc_id] = {
                        'document_id': doc_id,
                        'text': values["title"] + " " + values["text"] 
                    }
            self.collection = collection_
        else:
            self.collection = Collection

        return self.collection

    def get_model(self):
        return "BM25"

    def get_tokenizer(self):
     
        return self.tokenize
