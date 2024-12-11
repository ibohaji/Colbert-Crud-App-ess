from typing import List, Dict, TypedDict

class Query(TypedDict):
    query_id: str
    text: str

class Document(TypedDict):
    document_id: str
    text: str

class Result(TypedDict):
    query_id: str
    document_id: str
    score: float
    text: str  # Optional 