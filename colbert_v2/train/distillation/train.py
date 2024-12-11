import argparse
import csv
import json

from colbert import Trainer
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert.infra.run import Run
import os  



LR = 1e-05
WARMUP = 20_000
#################### ALL THESE CONVERSION FUNCTION ARE NEEDED CAUSE COLBERT IS INCONSISTENT WITH ITS INPUT FORMATS ####################
#################### AND CONTAIN A LOT OF BUGS IN ITS CODE ####################
#################### THESE FUNCTIONS ARE USED TO CONVERT THE INPUT DATA TO THE FORMAT COLBERT EXPECTS ####################
#################### NEED TO BE DELETED AND INTEGRATED INTO THE MAIN SCRIPT ####################

def convert_json_file_to_jsonl(input_json_path, output_jsonl_path):
    with open(input_json_path) as json_file:
        input_json = json.load(json_file)

    with open(output_jsonl_path, 'w') as jsonl_file:
        for key, value in input_json.items():
            jsonl_file.write(json.dumps({"qid": key, "question": value}) + '\n')



def json_to_tsv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        content = infile.read().strip()  # Read file and strip any surrounding whitespace

        if not content:
            raise ValueError(f"The input file {input_file} is empty or invalid.")

        data = json.loads(content)  # Use loads here as we have already read the content

    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile, delimiter='\t')

        for key, value in data.items():
            title = value.get('title', '').strip()
            text = value.get('text', '').replace('\n', ' ').strip()  # Remove newlines and trailing spaces
            combined_value = (title + " " + text).strip()  # Ensure there's a space between title and text

            if key and combined_value:  # Ensure both key and combined_value are non-empty
                writer.writerow([key, combined_value])
            else:
                print(f"Skipping entry with key: {key} due to missing title or text")


def convert_jsonl_with_scores(input_jsonl_path, output_jsonl_path):
    """Converts JSONL data from ["query_id", [[score, pid], ...]] to
    ["query_id", [pid1, pid2, ...], [score1, score2, ...]]
    """
    with open(input_jsonl_path, encoding='utf-8') as infile, \
         open(output_jsonl_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            try:
                query_id, pids_scores = json.loads(line)
                pids = [pair[1] for pair in pids_scores]
                scores = [pair[0] for pair in pids_scores]
                new_entry = [query_id, pids, scores]
                outfile.write(json.dumps(new_entry) + '\n')
                print(f'Sample of the converted file {output_jsonl_path}:')

            except Exception as e:
                print(f"Error processing line: {line}")
                print(e)


def convert_quirky_json(input_file, output_file):
    with open(input_file) as infile, open(output_file, 'w') as outfile:
        for line in infile:
            try:
                entry = json.loads(line.strip())
                query_id, pids_scores = entry

                formatted_entry = [query_id] + [[pid, score] for score, pid in pids_scores]
                outfile.write(json.dumps(formatted_entry) + '\n')

            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON on line: {line}. Error: {e}")
            except Exception as e:
                print(f"Error processing line: {line}. Error: {e}")

######################################################################################################################################
######################################################################################################################################
######################################################################################################################################




def run_distillation(triples_, queries_, collection_, experiment, nranks, bsize, doc_maxlen):

    with Run().context(RunConfig(nranks=nranks, experiment=experiment)):
        config = ColBERTConfig(bsize=bsize, checkpoint='colbert-ir/colbertv1.9', lr=LR, warmup=20_000, accumsteps=1, doc_maxlen=doc_maxlen, dim=128, attend_to_mask_tokens=False, nway=64, similarity='cosine', use_ib_negatives=True)
        trainer = Trainer(triples=triples_, queries=queries_, collection=collection_, config=config)
        trainer.train(checkpoint='colbert-ir/colbertv1.9')



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--triples_path', type=str, required=True)
    parser.add_argument('--queries', type=str, required=True)
    parser.add_argument('--collection', type=str, required=True)
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=os.getcwd())
    parser.add_argument('--nranks', type=int, default=1)
    parser.add_argument('--bsize', type=int, default=32)
    parser.add_argument('--doc_maxlen', type=int, default=240)
    parser.add_argument('--lr', type=float, default=1e-05)
    parser.add_argument('--nway', type=int, default=64)

    args = parser.parse_args()
    triples = args.triples_path
    queries = args.queries
    collection = args.collection
    experiment = args.experiment
    output_dir = args.output_dir
    nranks = args.nranks
    bsize = args.bsize
    doc_maxlen = args.doc_maxlen
    
    run_distillation(triples, queries, collection, experiment, nranks, bsize, doc_maxlen)

