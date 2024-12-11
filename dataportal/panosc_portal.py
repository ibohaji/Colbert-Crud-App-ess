from colbert_v2.models.searcher import ColBERTSearcher
from colbert_v2.models.indexer import ColBERTIndexer
from colbert_v2.config import Config
import json
import csv
import os

class PanoscPortalColbert:
    def __init__(self, documents_path=None, checkpoint='experiments/panson_colbert/none/2024-10/27/22.44.24/checkpoints/colbert', experiment='experiments', index_name='panosc'):
        self.colbert_config = Config()
        
        self.experiment = experiment
        self.index_name = index_name
        self.checkpoint = checkpoint
        self.searcher = ColBERTSearcher(self.index_name, "", self.checkpoint, self.experiment)
        
        if documents_path:
                if documents_path.endswith(".json"):
                    self.docs = self.load_json(documents_path)
                elif documents_path.endswith(".tsv"):
                    self.docs = documents_path
                else:
                    raise ValueError(f"File {documents_path} is not a valid json or tsv file")
            
                if not os.path.exists(self.index_name):
                    self._index_documents(self.docs)
    



    def load_json(self, collection_path: str):
        collection = []

        with open(collection_path, encoding="utf-8") as f:
            # loadd the json 
            data = json.load(f)

            for line_idx, (k, v) in enumerate(data.items()):

                pid, title, passage = line_idx, v['text'], v['title']
                passage = title + ' | ' + passage
                collection.append(passage)


            return collection

    def json_to_tsv(self, path: str):
        # read the .json file and convert and it as .tsv
        # Colbert, unfortunatly, does not support json files, so we need to convert them to tsv
        # This is a simple conversion, but it may not be the most efficient.
        # We read the pid, title, and text and store them as headerless csv files 
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # headerless csv file
            with open(path.replace(".json", ".tsv"), "w", encoding="utf-8") as f:
                for k,v in data.items():
                    # Removing extra spacing and newlines
                    cleaned_text = v['text'].replace("\n", " ").replace("\t", " ")
                    cleaned_title = v['title'].replace("\n", " ").replace("\t", " ")
                    f.write(f"{k}\t{cleaned_title}\t{cleaned_text}\n")
        
        return path.replace(".json", ".tsv")

    def search(self, query: str):
        try:
            ranking = self.searcher.search_query(query)
            return self.format_response(ranking)
        except Exception as e:
            print(f"Search query failed: {str(e)}")
            return {}
    
    def to_dictionary(self, ranking):
        """
        Converts the ranking to a dictionary with doc_id as keys and dictionaries containing rank and score as values.
        """
        json_rankings = {}
        for doc_id, rank, score in zip(ranking[0], ranking[1], ranking[2]):
            json_rankings[doc_id] = {'score': score, 'rank': rank}
        return json_rankings
    
    def format_response(self, results):
        """
        Formats the search results into a sorted list of dictionaries containing doc_id, rank, score, and title.
        """
        results = self.to_dictionary(results)
        response = sorted(
            [
                {
                    'doc_id': doc_id,
                    'rank': info['rank'],
                    'score': info['score'],
                    'title': self.docs.get(str(doc_id), {}).get('title', 'No Title')
                }
                for doc_id, info in results.items()
            ],
            key=lambda x: x['rank']
        )
        return response



    def _index_documents(self, documents: dict[dict]):
        indexer = ColBERTIndexer(self.colbert_config, documents, self.index_name, self.experiment, self.experiment)
        indexer.index_documents()


