# run_inference.py
# Run inference across all models 
from colbert_v2.config import Config
from inference.models import (
    BM25Retriever, ColBERTv2,
    Query, Document, Result,
    DenseRetrieval, Reranker
)
from inference.experiment import ExperimentTracker
import time
import torch
from typing import List, Union, Optional, Dict
from inference.data.data_loader import DataLoader
from pathlib import Path
import argparse
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from datetime import datetime




def evaluate_results(qrels, results):
    """
    Evaluate results using BEIR metrics.
    """
    evaluator = EvaluateRetrieval()
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, [10, 20, 100, 1000])
    
    print("\nRetrieval Metrics:")
    print("-" * 50)
    
    metrics = {}
    for k in [10, 20, 100, 1000]:
        print(f"NDCG@{k}: {ndcg[f'NDCG@{k}']:.4f}")
        print(f"Recall@{k}: {recall[f'Recall@{k}']:.4f}")
        print(f"Precision@{k}: {precision[f'P@{k}']:.4f}")
        print("-" * 25)
        
        if k in [1,10,100,1000]:
            metrics[f'NDCG@{k}'] = ndcg[f'NDCG@{k}']
            metrics[f'Recall@{k}'] = recall[f'Recall@{k}']
    
    return metrics

def run_experiment(
    model: Union[DenseRetrieval, Reranker],
    collection_path: str,
    queries_path: str,
    device: str
) -> None:
    experiment = ExperimentTracker()
    model_name = model.__class__.__name__
    
    # First run on GPU
    print(f"\nRunning on {device.upper()}...")
    if hasattr(model, 'load_model'):
        model.load_model(device)
    
    retrieval_model_name = None
    if hasattr(model, 'retrieval_model'):
        retrieval_model_name = model.retrieval_model.__class__.__name__
    
    experiment.start_experiment(model_name, device, retrieval_model_name)
    
    model.load_collection(collection_path)
    queries = model.load_queries(queries_path)
    if model.collection is None:
        raise ValueError(f"Failed to load collection from {collection_path}")
    if model.queries is None:
        raise ValueError(f"Failed to load queries from {queries_path}")
    
    start_time = time.time()
    model.create_index()
    experiment.experiments[model_name]["index_time"] = time.time() - start_time
    
    experiment.experiments[model_name]["model_dimension"] = model.get_model_dimension_embeddings()
    experiment.experiments[model_name]["index_size_mb"] = model.get_index_size()
    experiment.experiments[model_name]["total queries"] = len(queries)
    experiment.experiments[model_name]["total docs"] = len(model.collection)
    
    start_time = time.time()
    end_tracking = experiment.track_search(model_name)
    results = model.search_all(queries)
    end_tracking()
    
    #timestamp = model.save_results(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment.export_results(model_name, timestamp)
    
    if hasattr(model, 'load_model'):
        model.load_model(device)
        
        start_time = time.time()
        end_tracking = experiment.track_search(model_name)
        results = model.search_all(queries)
        end_tracking()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        experiment.export_results(model_name, timestamp)
    else:
        print("\nGPU not available, skipping GPU run")
    
    print("\n" + "="*50)
    print(f"{model_name.upper()} EXPERIMENT RESULTS for {device.upper()}")
    print("="*50)
    
    exp_data = experiment.experiments[model_name]
    
    print("\nModel Characteristics:")
    print(f"• Embedding dimension: {exp_data['model_dimension']}")
    print(f"• Index size: {exp_data['index_size_mb']:.2f} MB")
    
    print("\nPerformance Metrics:")
    print(f"• Average latency: {experiment.calculate_average_latency(model_name)} seconds")
    print(f"• Throughput: {experiment.calculate_throughput(model_name)} queries/second")
    print(f"• Average memory: {experiment.calculate_average_memory(model_name)} MB")
    
    print("\nInitialization Times:")
    print(f"• Load time: {exp_data['load_time']:.6f} seconds")
    print(f"• Warmup time: {exp_data['warmup_time']:.6f} seconds")

    return results

def load_qrels(qrels):
    
    qrels_dict = {}
    with open(qrels, 'r') as file:
        next(file)
        for line in file:
            query_id, doc_id, relevance = line.strip().split('\t')
            if query_id not in qrels_dict:
                qrels_dict[query_id] = {}
            qrels_dict[query_id][doc_id] = int(relevance)
    return qrels_dict

def main(collection_path: str, queries_path: str, qrels_path: str, device:str):

    
    project_root = Path(__file__).parent
    queries_path = str(project_root / queries_path)
    collection_path = str(project_root / collection_path)
    qrels = load_qrels(str(project_root / qrels_path))


    print(f"Loading from paths:\nCollection: {collection_path}\nQueries: {queries_path}")
    
    print(f"Running on {device.upper()}...")
    # Initialize models
    #bert_retriever = BERTRetriever(
     #   model_name="sentence-transformers/msmarco-bert-base-dot-v5",
   # device=device
#)
    #  results = run_experiment(bert_retriever, collection_path, queries_path, device)

 #   bi_encoder = BiEncoder(
  #  model_name="sentence-transformers/msmarco-bert-base-dot-v5",
  #  device=device
#)

 #   cross_encoder = CrossEncoder(
  #  model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
   # retrieval_model=bi_encoder,
    #device=device
#)

    #results = run_experiment(cross_encoder, collection_path, queries_path, device)

    
   # bm25 = BM25Retriever()
   # bm25.es_path = '/zhome/3a/7/145702/elasticsearch-8.5.0'  
   # print("Running experiment on Elasticsearch")
   # results = run_experiment(bm25, collection_path, queries_path, device)
   # print(dict(list(results.items())[:1]))  # Convert back to a dictionary for a cleaner output        evaluate_results(qrels, results)
   # evaluate_results(qrels, results)

    config = Config()
    colbert = ColBERTv2(
        config=config,
        device=device,
        collection_path=collection_path
)

    results = run_experiment(colbert, collection_path, queries_path, device)
    evaluate_results(qrels, results)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference experiments")
    parser.add_argument("--collection_path", type=str, required=True, help="Path to the collection file")
    parser.add_argument("--queries_path", type=str, required=True, help="Path to the queries file")
    parser.add_argument("--qrels", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    args = parser.parse_args()

    main(args.collection_path, args.queries_path, args.qrels, device=args.device)