# retriever.py 
# initial retriever for the collection, intended to be reranked 

from abc import ABC, abstractmethod

class Retriever(ABC):

    @abstractmethod
    def retrieve_top_k(self, query):
        pass   

    @abstractmethod
    def create_index(self):
        pass

    @abstractmethod
    def get_index_size(self):
        pass

    def __repr__(self):
        return self.__class__.__name__

    def load_collection(self, collection_path: str):
        pass

    def load_queries(self, queries_path: str):
        pass

