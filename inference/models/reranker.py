from abc import ABC, abstractmethod
from typing import List, Tuple
from pathlib import Path
from datetime import datetime
from .data_types import Query, Document, Result

class Reranker(ABC):
    def __init__(self, device):
        self.device = device
    
    @abstractmethod
    def rerank(self, query: Query, candidates: List[Result]) -> List[Result]:
        pass

    @abstractmethod
    def load_model(self, device: str):
        pass

    def rank(self, results: List[Result]) -> List[Tuple[str, str, int]]:
        """Convert results to ranking format (qid, pid, rank)
        This method expects results to be ranked by score in descending order
        """
        rankings = []
        for rank, result in enumerate(results, start=1):
            rankings.append((
                result['_id'],
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