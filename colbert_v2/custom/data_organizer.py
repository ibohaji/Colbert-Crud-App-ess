import json
import logging
from typing import Dict
import json
import csv

class GenQueryData:
    def __init__(self, generated_queries_path, qrels_path = None):
        self.genqueries = generated_queries_path
        self.qrels = qrels_path
        self.queries_dict = self.load_data()

    def load_data(self):

        if self.genqueries.endswith('.tsv'):
            return self.load_tsv(self.genqueries)
        q_map_dict = {}
        with open(self.genqueries) as f:
            queries = json.load(f)
        if self.qrels :
            with open(self.qrels) as f:
                qrels = json.load(f)

            for qid,pid in qrels.items():
                q_map_dict[qid] = { "text": queries[qid], "doc_id": pid }

            return q_map_dict
        
        for qid, text in queries.items():
            q_map_dict[qid] = { "text": text }


        return q_map_dict

    def load_tsv(self, input_file:str):
        data_dict = {}
        with open(input_file, 'r', newline='', encoding='utf-8') as f:
            for line in f:
                qid, text = line.strip().split('\t', 1)
                data_dict[qid] = {'text': text}
            
            if self.qrels:
                with open(self.qrels) as f:
                    qrels = json.load(f)
                    for qid,pid in qrels.items():
                     if qid in data_dict:    
                        data_dict[qid] = { "text": data_dict[qid]['text'], "doc_id": pid }


        return data_dict



class CollectionData:
    def __init__(self, collection_path):
        self.collection_path = collection_path
        self.collection_dict = list(self.load_collection(collection_path))
        self.collection_pids = self.make_collection_pids()
        logging.info(f"Loaded {len(self.collection_dict)} documents from '{collection_path}'.")


    def load_collection(self, collection_path):
        if collection_path.endswith('.jsonl'):
            return self.load_jsonl(collection_path)
        elif collection_path.endswith('.json'):
            return self.load_json(collection_path)
        elif collection_path.endswith('.tsv'):
            return self.load_tsv(collection_path)
        else:
            raise ValueError("Input file must be a .json or .jsonl file")

    def load_jsonl(self, path):
        """Loads documents from a JSONL file."""
        if path.endswith('.jsonl'):
                
            with open(path, encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        yield {
                            '_id': entry['_id'],
                            'text': entry['text'],
                            'title': entry['title']
                        }
                    except json.JSONDecodeError as e:
                        logging.error(f"Error decoding JSON line: {e}")
                        



    
    def load_json(self, input_file:str):
        # load it as a list 
        data = []
        with open(input_file) as f:
            json_data = json.load(f)
        
        for key, doc in json_data.items():
            # lazy loader
            doc['_id'] = key
            doc['text'] = doc.get('title', '') + doc.get('text', '')
            yield doc
    
    def load_jsonl(self, input_file:str):
            if input_file.endswith('.jsonl'):
                    
                with open(input_file, encoding='utf-8') as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            yield {
                                '_id': entry['_id'],
                                'text': entry['text'],
                                'title': entry['title']
                            }
                        except json.JSONDecodeError as e:
                            logging.error(f"Error decoding JSON line: {e}")
                            

    


    def make_tsv(self, output_file:str):
            with open(self.collection_path, 'r') as infile, open(self.collection_path, 'w', newline='', encoding='utf-8') as outfile:
                data = json.load(infile)

                tsv_writer = csv.writer(outfile, delimiter='\t')

                tsv_writer.writerow(["_id", "text"])

                for key, value in data.items():
                    title_text = f"{value['title']} {value['text']}"
                    tsv_writer.writerow([key, title_text])
                

    def make_collection_pids(self):
        """Creates a list of all document IDs."""
        return [doc['_id'] for doc in self.collection_dict]


