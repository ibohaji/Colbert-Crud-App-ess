import argparse
import logging
import tqdm
import ujson
from colbert_v2 import CollectionData, GenQueryData
from .sampler import HardNegativesSampler
import os 
from .el_search import EsSearcher
import json

def save_pairs(scored_triples, output_path):
    with open(output_path, 'w') as f:
        for triple in scored_triples:
            ujson.dump(triple, f)
            f.write('\n')
    print(f"Saved pairs to {output_path}")


def integrate_scores(triples, scores):
    return [(query_id, pos_id, neg_id, score) for (query_id, pos_id, neg_id), score in zip(triples, scores)]



def generate_pairs(queries_data, hard_negatives):
    qids = []
    pids = []

    for query_id, query_data in tqdm.tqdm(queries_data.queries_dict.items()):
        positive_doc_id = query_data['doc_id']
        negatives = hard_negatives.get(query_id, [])

        qids.append(query_id)
        pids.append(positive_doc_id)

        for negative_doc_id in negatives:
            qids.append(query_id)
            pids.append(negative_doc_id)

    return qids, pids


def main(generated_queries_path, collection_path, out_path, qrel, num_negatives=3):
    print('starting...')
    with EsSearcher() as es_searcher:

        queries_data = GenQueryData(generated_queries_path, qrel)
        collection_data = CollectionData(collection_path)
        hard_negatives = HardNegativesSampler(queries=queries_data, collection=collection_data, es=es_searcher ).get_hard_negatives_all(num_negatives=num_negatives)
        qids, pids = generate_pairs(queries_data, hard_negatives)


    save_pairs(qids, os.path.join(out_path,'qids.json'))
    save_pairs(pids, os.path.join(out_path,'pids.json'))
    print("saved pairs")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()

    parser.add_argument("--generated_queries_path",
                         type=str, required=True, 
                         help="Path to the generated queries"
                         )
    
    parser.add_argument("--collection_path",
                          type=str, 
                          required=True, 
                          help="Path to the collection"
                          )
    
    parser.add_argument("--num_negatives",
                         default=64,
                         type=int, 
                         help="Number of negative samples to retrieve from Elasticsearch"
                         )
    
    parser.add_argument('--qrel_path',
                        type=str,
                        help="Path to the qrel file"
                        )
    
    parser.add_argument('--out_path',
                        help="Output directory of the pairs"
                        )
    args = parser.parse_args()
    print('starting..')

    main(args.generated_queries_path, args.collection_path, args.out_path, args.qrel_path, args.num_negatives)
