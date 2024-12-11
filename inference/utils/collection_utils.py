from typing import Dict, List, Tuple

def standardize_collection_ids(collection: Dict) -> Dict:
    """
    Rewrite collection document IDs to match their line index (0-based).
    Returns a new collection with standardized IDs and maintains original ID mapping.
    standardiezed : {str_idx: {'document_id': str_idx, 'text': doc['text']}}
    id_mapping : {original_id: str_idx}
    reverse_mapping : {str_idx: original_id}
    """
    standardized = {}
    id_mapping = {}  
    reverse_mapping = {}
    
    for idx, (original_id, doc) in enumerate(collection.items()):
        str_idx = str(idx)
        id_mapping[original_id] = str_idx
        reverse_mapping[str_idx] = original_id
        
        standardized[str_idx] = {
            'document_id': str_idx,
            'text': doc['text']
        }
    
    return standardized, id_mapping, reverse_mapping

def map_back_results(results: List[Tuple[str, str, int]], reverse_mapping: Dict) -> List[Tuple[str, str, int]]:

    mapped_results = {}
    # result.tolist() is [(qid,doc_id, score),...]
    for qid, doc_id, rank, score in results.tolist():
        if qid not in mapped_results:
            mapped_results[qid] = {}

        mapped_results[qid][reverse_mapping[str(doc_id)]] = score

    return mapped_results