# random_runs.py
from colbert_v2.config import Config
from inference.models import (
    BM25Retriever, ColBERTv2,
    Query, Document, Result
)
from beir.retrieval.evaluation import EvaluateRetrieval
from inference.data.data_loader import QueryLoader, CollectionLoader
from inference.experiment import ExperimentTracker
import time
import torch
from typing import List, Union, Optional, Any
from run_inference import run_experiment
from pathlib import Path
import random
import argparse
import json
from pprint import pprint
import os



def load_qrels_json(qrels):
    qrels_formatted={}
    with open(qrels, "r") as f: 
        qrels_dict = json.load(f)
        for key, value in qrels_dict.items():

            if str(key) not in qrels_formatted:
                qrels_formatted[str(key)] = {}

            qrels_formatted[str(key)][str(value)] = int(1)

    return qrels_formatted


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



def evaluate_results(qrels, results):
    """
    Evaluate results using BEIR metrics.
    """
    qrels = load_qrels_json(qrels)
    evaluator = EvaluateRetrieval()
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, [1, 10, 100, 1000])
    
    print("\nRetrieval Metrics:")
    print("-" * 50)
    
    metrics = {}
    for k in [1, 10, 100, 1000]:
        print(f"NDCG@{k}: {ndcg[f'NDCG@{k}']:.4f}")
        print(f"Recall@{k}: {recall[f'Recall@{k}']:.4f}")
        print(f"Precision@{k}: {precision[f'P@{k}']:.4f}")
        print("-" * 25)
        
        if k in [1,10,100,1000]:
            metrics[f'NDCG@{k}'] = ndcg[f'NDCG@{k}']
            metrics[f'Recall@{k}'] = recall[f'Recall@{k}']
    
    return metrics


class ModelConfig:
    """Configuration container for model initialization parameters"""
    def __init__(self, model_class, **kwargs):
        self.model_class = model_class
        self.params = kwargs

class ModelFactory:
    """Factory class for creating model instances with state persistence"""
    _instances = {}  # Class variable to store model instances

    @classmethod
    def create_model(cls, config: ModelConfig, force_new: bool = False) -> Any:
        model_name = config.model_class.__name__
        
        # Return existing instance if available and not forced to create new
        if not force_new and model_name in cls._instances:
            return cls._instances[model_name]
        
        # Create new instance
        instance = config.model_class(**config.params)
        cls._instances[model_name] = instance
        return instance

    @classmethod
    def clear_instances(cls):
        """Clear all cached model instances"""
        cls._instances.clear()

model_configs = {
  
    "Lucene bm25": ModelConfig(
        BM25Retriever),
        "ColBERTv2": ModelConfig(
       ColBERTv2,
        config = Config(),
        device = "cuda"
)
                
    
}



def save_evaluaton(evaluation, model, run:int):
    out_dir = os.path.join("inference", "measured_data", "sub_runs","panosc", model)
    print(f"saving evaluations to {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    file = os.path.join(out_dir, f"{run}.json")
    with open(file, "w") as f:
        json.dump(evaluation, f)


def run_supsamples(model, queries: str, collection_path: str):
    experiment = ExperimentTracker()
    model_name = model.__class__.__name__


    retrieval_model_name = None
    if hasattr(model, 'retrieval_model'):
        retrieval_model_name = model.retrieval_model.__class__.__name__
    
    experiment.start_experiment(model_name, "gpu", retrieval_model_name)
    model.load_queries(queries)
  
    if model.queries is None:
        raise ValueError(f"Failed to load queries from {queries_path}")
    


    experiment.experiments[model_name]["model_dimension"] = model.get_model_dimension_embeddings()
    experiment.experiments[model_name]["index_size_mb"] = model.get_index_size()
    experiment.experiments[model_name]["total queries"] = len(queries)
    experiment.experiments[model_name]["total docs"] = len(model.collection)
    
    results = model.search_all(queries)
    
    
    print("\n" + "="*50)
    print(f"{model_name.upper()} EXPERIMENT RESULTS for GPU")
    print("="*50)
    
    exp_data = experiment.experiments[model_name]
    
    print("\nModel Characteristics:")
    print(f"• Embedding dimension: {exp_data['model_dimension']}")
    print(f"• Index size: {exp_data['index_size_mb']:.2f} MB")
    

    return results

def run(dataset_path: str, queries_path: str, qrel_path):

    factory = ModelFactory()
    seed = 42
    n = 10
    
    Queries = QueryLoader(queries_path).to_dict()
    
    # Initialize models once before the loop
    models = {
        model_name: factory.create_model(config)
        for model_name, config in model_configs.items()
    }
    
    # Load and index models once
    for model_name, model in models.items():
        print(f"Initializing model: {model_name}")
        if not model.indexed:
            model.load_collection(dataset_path)
            model.load_model("cuda")
            model.create_index()
    
    for i in range(n):
        print(f"Running experiment {i+1} of {n}")
        subset_keys = random.sample(list(Queries.keys()), 250)
        subset_queries = {key: Queries[key] for key in subset_keys}
        
        for model_name, model in models.items():
            print(f"Running inference with model: {model_name}")
            results = run_supsamples(model, subset_queries, dataset_path)   
            metrics = evaluate_results(qrels, results)
            save_evaluaton(metrics, model_name, i)

    # Optionally clear model instances after all experiments
    factory.clear_instances()
    
if __name__ == '__main__':

    queries_path="data/panson_data/validation_queries_cleaned.tsv"
    collection_path="data/panson_data/panson_docs_summaries.json"
    qrels="data/panson_data/qrel.json"

    from multiprocessing import freeze_support
    freeze_support()


    project_root = Path(__file__).parent

    model_configs["ColBERTv2"].params["collection_path"] = collection_path

    run(collection_path, queries_path, qrels)
