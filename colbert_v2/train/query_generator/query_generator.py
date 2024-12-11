import argparse
import json
import os
from collections import defaultdict
from itertools import count
from typing import Dict
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from ...config import MetaData


class QueryGenerator:
    def __init__(self, model_name:str, input_documents, output_path = None)->None:
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_path = self.create_directory(output_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.documents =  self.index_document(input_documents) 
        self.counter = count()
    


    def create_directory(self, output_path:str)->str:
        if output_path is None:
            output_path = os.path.join(os.getcwd(), "generated_queries")
        os.makedirs(output_path, exist_ok=True)
        return output_path
    
    def load_jsonl(self, input_file:str)->Dict:
        data = []
        with open(args.input_documents) as f:
            for line in f:
                doc = json.loads(line)
                data.append(doc)
        data = self.index_document(data)
        return data
    
    def load_json(self, input_file:str)->Dict:
        # load it as a list 
        data = []
        with open(input_file) as f:
            json_data = json.load(f)
        
        for key, doc in json_data.items():
            doc['id'] = key
            data.append(doc)
        return data
    

    def _load_documents(self, input_documents:str)->Dict:

        if input_documents.endswith('.jsonl'):
            data = self.load_jsonl(input_documents)

        elif input_documents.endswith('.json'):
            data = self.load_json(input_documents)

        elif input_documents.endswith('.tsv'):
            raise NotImplementedError("Tsv file not supported yet")
        
        else:
            raise ValueError("Input file must be a .json or .jsonl file")
        
        return data

    def index_document(self, documents:Dict)->Dict:

        if isinstance(documents, str):
            documents = self._load_documents(documents)
        
        return { idx: doc['title'] + doc['text'] for idx, doc in enumerate(documents) }

    def generate_query_ids(self, _range):
        return [next(self.counter) for _ in range(_range)]

    @classmethod
    def _removeNonAscii(s:str) -> str : return "".join(i for i in s if ord(i) < 128)


    def generate_split_queries(self, generated_queries: Dict[int, str], qrel: Dict[int, int], hold_out_size: int = 2):
        
        train_queries = {}
        validation_queries = {}

        queries_by_doc = defaultdict(list)
        for query_id, doc_id in qrel.items():
            queries_by_doc[doc_id].append(query_id)
        
        for doc_id, query_ids in queries_by_doc.items():
            if len(query_ids) <= hold_out_size + 1:
                validation_queries.update({qid: generated_queries[qid] for qid in query_ids})
            else:
                validation_queries.update({qid: generated_queries[qid] for qid in query_ids[:hold_out_size]})
                train_queries.update({qid: generated_queries[qid] for qid in query_ids[hold_out_size:]})
        
        validation_output_file = os.path.join(self.output_path, "validation_queries.tsv")
        train_output_file = os.path.join(self.output_path, "train_queries.tsv")
        
        self.save_queries_to_tsv(validation_queries, validation_output_file)
        self.save_queries_to_tsv(train_queries, train_output_file)

        print(f"TSV files saved: {len(train_queries)} training queries and {len(validation_queries)} validation queries.")
        
    def save_queries_to_tsv(self, queries: Dict[int, str], output_file: str):
    
        with open(output_file, 'w', encoding='utf-8') as f:
            for qid, qtext in queries.items():
                f.write(f"{qid}\t{qtext}\n")
        
    def remove_duplicates(self, generated_queries:list):
        return list(set([query.lower() for query in generated_queries]))

    def save_queries_to_json(self, generated_queries, output_file):
        # Save generated queries to a JSON file
        with open(output_file, "w") as f:
            json.dump(generated_queries, f, indent=2)

    def save_queries_list(self, generated_queries, output_file):
        raise NotImplementedError

    def process_documents(self, document_folder, output_file):
        raise NotImplementedError

    def _clean_text(self, text):
        """Utility function to clean text by removing tabs and non-ASCII characters."""
        return text.replace("\t", " ").encode("ascii", "ignore").decode().strip()


    def generate_queries(self,batch_size:int = 16)->None:
        generated_queries = {}
        qrel = defaultdict(list)
        doc_items = [(doc_id, doc) for doc_id, doc in self.documents.items()]

        for start_idx in tqdm(range(0,len(doc_items),batch_size), desc="Generating queries"):
            batch_docs = doc_items[start_idx:start_idx+batch_size]

            for doc_id, doc in batch_docs:
                    doc = self._clean_text(doc)
                    input_ids = self.tokenizer.encode(
                        doc,
                        max_length=512,
                        truncation=True,
                        return_tensors='pt'
                        ).to(self.device)

                    outputs = self.model.generate(
                        input_ids=input_ids,
                        max_length=64,
                        do_sample=True,
                        top_p=0.95,
                        num_return_sequences=5
                    )

                    queries = [
                        self._clean_text(self.tokenizer.decode(output, skip_special_tokens=True)) 
                        for output in outputs
                        ]   
                    
                    queries = self.remove_duplicates(queries)
                    queries_id = self.generate_query_ids(len(queries))
                    
                    for _qid, query_text in zip(queries_id, queries):
                        generated_queries[_qid] = query_text
                        qrel[_qid] = doc_id
                        
        print(f"Total generated queries: {len(generated_queries)}")
        self.save_queries_to_json(generated_queries, f"{self.output_path}/generated_queries.json")
        self.save_queries_to_json(qrel, f"{self.output_path}/qrel.json")
        self.generate_split_queries(generated_queries, qrel)
        return generated_queries


        
if __name__ =="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_documents",
                         type=str,
                         required=True,
                         help="Folder containing the documents"
                           )
    parser.add_argument("--output_file",
                         type=str,
                         default=None,
                         help="Folder to store the documents"
                         )
    parser.add_argument("--model_name",
                         type=str,
                         default='doc2query/S2ORC-t5-base-v1',
                         help="Folder containing the documents"
                         )
    parser.add_argument("--top_p",
                        type=int,
                        default=0.95
                        )
    parser.add_argument("--num_genq",
                        type=int,
                        default=5,
                        help="Number of queries to generate per document"
                        )

    args = parser.parse_args()
    Qgen = QueryGenerator(args.model_name, args.input_documents, args.output_file)

    MetaData().update(top_p = args.top_p, num_genq = args.num_genq)
    Qgen.generate_queries()
