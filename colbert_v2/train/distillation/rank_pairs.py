import argparse
import json
from collections import defaultdict

import tqdm
import ujson
from colbert.infra import Run
from colbert.infra.config import RunConfig

from ...custom.data_organizer import CollectionData, GenQueryData
from .scorer import Scorer


def main(qids, pids, collection, queries):
    with Run().context(RunConfig(nranks=1)):

        scorer = Scorer(queries=queries, collection=collection)
        distillation_scores = scorer.launch(qids, pids)

    scores_by_qid = defaultdict(list)
    for qid, pid, score in tqdm.tqdm(zip(qids, pids, distillation_scores)):
        scores_by_qid[qid].append((score, pid))

    with open('distillation_scores.json', 'w') as f:
        for qid in tqdm.tqdm(scores_by_qid):
            # Reverse the order in the (score, pid) tuple to (pid, score)
            formatted_entry = [qid] + [[pid, score] for score, pid in scores_by_qid[qid]]
            f.write(json.dumps(formatted_entry) + '\n')

    output_path = f.name
    print(f"Saved distillation scores to {output_path}")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--qid_path', type=str, required=True)
    parser.add_argument('--pid_path', type=str, required=True)
    parser.add_argument('--collection_path', type=str, required=True)
    parser.add_argument('--queries_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()


    queries = GenQueryData(args.queries_path, qrels_path=None)
    collection = CollectionData(args.collection_path)

    queries = queries.queries_dict
    queries = {qid: query['text'] for qid, query in queries.items()}
    # save queries as a dictionary with qid as key and query text as value
    with open('queries_generated_mapping.json', 'w') as f:
        f.write(json.dumps(queries, indent=4))

    collection = collection.collection_dict
    collection = {doc['_id']: doc['text'] for doc in collection}

    with open(args.qid_path) as f:
        qids = [line.strip().strip('"') for line in f.readlines()]

    with open(args.pid_path) as f:
        pids = [line.strip().strip('"') for line in f.readlines()]


    main(qids, pids, collection, queries)


