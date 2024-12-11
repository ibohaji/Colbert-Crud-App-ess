# config.py
#from colbert.infra import ColBERTConfig
import json
from dataclasses import dataclass
from datetime import datetime
import hashlib
import os 
import random


@dataclass
class Config:
    CHECKPOINT = "colbert-ir/colbertv2.0"
    DOC_MAXLEN = 300
    NBITS = 2
    KMEANS_NITERS = 8  # KMeans iterations for FAISS clustering
    NCELLS = 4000  # Number of cells for coarse search
    NDOCS = 1000  # Number of documents to retrieve
    INDEX_NAME = "crud.colbert.index.inference"
    COLLECTION_PATH = "colbert-datasets/scifact/collection.tsv"
    QUERIES_PATH = "colbert_v2/data/queries.tsv"


@dataclass
class MetaData:
    """Class to store all the parameters of the experiments"""
    NWAY: int
    BSIZE: int
    NDCG: float
    SAMPLING_METHOD: str
    LR: float
    WARMUP: int
    BASE_MODEL: str
    EPOCHS: int
    NUM_QUERIES: int
    NUM_COLLECTION: int
    GPU_TYPE: str
    NUM_GPUS: int
    CHECKPOINT_PATH: str
    TRAINING_TIME: float
    DISTILLATION_TIME: float
    EVALUATION_TIME: float
    SEED: int


class Singleton(type):
    _instances = {}
    def __call__(cls):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__()
        return cls._instances[cls]


class MetaData(metaclass=Singleton):

    def __init__(self):
        self.EXPERIMENT_ID = self.generate_unique_id()
        self.NWAY = None
        self.BSIZE = None
        self.NDCG = None
        self.SAMPLING_METHOD = ''
        self.LR = None
        self.WARMUP = None
        self.BASE_MODEL = ''
        self.EPOCHS = None
        self.top_p =None
        self.num_genq = None
        self.NUM_QUERIES = None
        self.NUM_COLLECTION = None
        self.GPU_TYPE = ''
        self.NUM_GPUS = None
        self.CHECKPOINT_PATH = ''
        self.TRAINING_TIME = None
        self.DISTILLATION_TIME = None
        self.EVALUATION_TIME =None
        self.SEED = None
        self.GPU_TYPE = None
        self.gpu_utilization = []



    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.save_results()
        
    def generate_unique_id(self):
        """Generate a unique id based on current time and a random number"""
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        random_number = random.randint(1000, 9999)
        return f"{current_time}_{random_number}"

    def save_results(self):
        """Append the metadata to a JSON file dynamically"""
        data = self.__dict__
        base_path = os.path.join(os.getcwd(), 'experiment_results')

        run_path = os.path.join(base_path, self.EXPERIMENT_ID)
        os.makedirs(run_path, exist_ok=True)
        file_name = os.path.join(run_path, f'experiment_{self.EXPERIMENT_ID}_data.json')

        with open(file_name + '.json', 'w') as f:
            f.write(json.dumps(data, indent=4))

        print(f"MetaData saved to {file_name}")



