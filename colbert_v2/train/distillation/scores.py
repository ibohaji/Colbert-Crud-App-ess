import argparse

import ujson
from colbert.distillation.scorer import Scorer
from colbert_v2.custom.data_organizer import CollectionData, GenQueryData


def save_triples(scored_triples, output_path):
    with open(output_path, 'w') as f:
        for triple in scored_triples:
            ujson.dump(triple, f)
            f.write('\n')
    print(f"Saved triples to {output_path}")

def integrate_scores(triples, scores):
    return [(query_id, pos_id, neg_id, score) for (query_id, pos_id, neg_id), score in zip(triples, scores)]


def compute_scores(triples, queries_data, collection_data):
    queries = {qid: query['text'] for qid, query in queries_data.queries_dict.items()}
    collection = {doc['_id']: doc['text'] for doc in collection_data.collection_dict}

    scorer = Scorer(queries, collection, model='cross-encoder/ms-marco-MiniLM-L-6-v2')

    qids = [triple[0] for triple in triples]
    pids = [triple[1] for triple in triples]

    scores = scorer.launch(qids, pids)
    return scores


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--triples_path', type=str, required=True)
    parser.add_argument('--collection_path', type=str, required=True)


    args = parser.parse_args()
    gen_queries = GenQueryData(args.triples_path)
    collection_data = CollectionData(args.collection_path)
    triples = args.triples
    scores = compute_scores(triples, gen_queries, collection_data)
    scored_triples = integrate_scores(triples, scores)

    save_triples(scored_triples, 'scored_triples_.json')
