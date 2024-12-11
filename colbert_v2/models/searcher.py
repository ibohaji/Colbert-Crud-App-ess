# models/searcher.py
import argparse
import json
import os
from time import time
from ..custom.execution_monitor import monitor_gpu
from colbert import Searcher
from colbert.data import Queries
from colbert.infra import ColBERTConfig, Run, RunConfig
from ..config import Config, MetaData


class ColBERTSearcher:
    def __init__(self, index_name, queries_path, checkpoint, experiment):
        self.config = Config()
        self.config.INDEX_NAME = index_name
        self.queries_path = queries_path
        self.checkpoint = checkpoint
        self.experiment = experiment
    
    def load_model(self):
        config = ColBERTConfig(root='experiments')
        self.searcher = Searcher(index=self.config.INDEX_NAME, config=config, checkpoint = self.checkpoint)

    def search_query(self, query):
        with Run().context(RunConfig(nranks=1, experiment='experiments')):
           # config = ColBERTConfig(root='experiments')
           # searcher = Searcher(index=self.config.INDEX_NAME, config=config, checkpoint = self.checkpoint)
            start_time = time()
            ranking = self.searcher.search(query, k=100)
        total_time = time() - start_time
        print(f"Search completed successfully in {total_time} seconds")
        MetaData().update(Search_time=total_time)
        return ranking

    
    @monitor_gpu
    def search(self, ranking_output):
        start_time = time()
        with Run().context(RunConfig(nranks=1, experiment='experiments')):
            start_time = time()
            queries = Queries(self.queries_path)
            ranking = self.searcher.search_all(queries, k=1000)
            os.makedirs(ranking_output, exist_ok=True)
            ranking.save(ranking_output)

        total_time = time() - start_time
        print(f"Search completed successfully in {total_time} seconds")
        MetaData().update(Search_time=total_time)
        
if __name__ == "__main__":
    ## parse args and call searcher
    parser = argparse.ArgumentParser()
    parser.add_argument('--Queries', type=str, default="crud.colbert.index")
    parser.add_argument('--output_path', type=str, help='Path to store rankings')
    parser.add_argument('--experiment', type=str, default="scifact")
    parser.add_argument('--index_name', type=str, default="experiments")
    parser.add_argument('--checkpoint', type=str, default="experiments")
    args = parser.parse_args()

    
    queries = args.Queries
    output = args.output_path
    checkpoint=args.checkpoint
    experiment=args.experiment
    config = Config()
    searcher = ColBERTSearcher(args.index_name, queries, checkpoint, experiment)
    searcher.search(output)