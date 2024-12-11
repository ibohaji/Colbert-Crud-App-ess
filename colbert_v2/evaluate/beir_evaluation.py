import argparse
import csv
import json
import logging
import random
import sys
import os
import pathlib
from beir.retrieval.evaluation import EvaluateRetrieval
from ..config import MetaData

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

csv.field_size_limit(sys.maxsize)

def load_qrel(filepath):
    with open(filepath, "r") as f:
        raw_qrels = json.load(f)

    qrels = {query_id: {str(doc_id): 1} for query_id, doc_id in raw_qrels.items()}
    
    return qrels

def load_rankings(filepath):
    """Load rankings from a TSV file."""
    results = {}
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            qid, doc_id, rank = row[0], row[1], int(row[2])
            if qid not in results:
                results[qid] = {}
            results[qid][doc_id] = 1 / (rank + 1)

    return results

def main(qrel_path, rankings_path, k_values, output_dir):
    qrels = load_qrel(qrel_path)
    results = load_rankings(rankings_path)
    print(f"qrel example: {list(qrels.items())[:10]}")
    print(f"results example: {list(list(results.items())[:10])[:5]}")
    evaluator = EvaluateRetrieval()
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values)
    mrr = evaluator.evaluate_custom(qrels, results, k_values, metric='mrr')

    #### Log results and sample output ####
    colbert_metrics = {
        "NDCG": ndcg,
        "MAP": _map,
        "Recall": recall,
        "Precision": precision,
        "MRR": mrr,
    }

    # Store the results in MetaData
    MetaData().update(colbert_metrics=colbert_metrics)


    directory = pathlib.Path(output_dir) / "colbert_metrics.json"
    with open(directory, "w") as f:
        json.dump(colbert_metrics, f, indent=2)

    logging.info("Metrics have been saved to colbert_metrics.json.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--qrel_path', type=str, required=True, help="Path to the qrel JSON file.")
    parser.add_argument('--rankings_path', type=str, required=True, help="Path to the rankings TSV file.")
    parser.add_argument('--k_values', nargs='+', type=int, default=[1,3,5,10,100,1000])
    parser.add_argument('--output_dir', type=str, default=os.getcwd())
    args = parser.parse_args()

    main(args.qrel_path, args.rankings_path, args.k_values, args.output_dir)
