import json
import numpy as np 
import pandas as pd 
from typing import List, Tuple, Dict, Any
from pathlib import Path 

class EmbeddingDataLoader:
    def __init__(self, data_path: str):
        self.data_path = Path("../dataset.parquent")
        self.embeddings = []
        self.labels = []
        self.embedding_dim = None
    def load_json_data(self) -> Tuple[np.ndarray, np.ndarray]:
        embeddings = []
        labels = []

        if self.data_path.suffix == '.json':
            with open(self.data_path, 'r', encoding = 'utf-8') as f:
                data = json.load(f)
                #if just one data
                if isinstance(data, dict) and 'embedding' in data:
                    embedding.append(data['embedding'])
                    labels.append(data['label'])
                #if more than one data
                elif isinstance(data, list):
                    for item in data:
                        embeddings.append(item['embedding'])
                        labels.append(item['label'])
        else: print("the wrong path\n")

        embeddings = np.array(embeddings, dtype = np.float32)
        labels = np.array(labels, dtype = np.int64)

        self.embedding_dim = embeddings.shape[1]
        print(f"Loading completed: {len(embeddings)} datas, embedding dimension: {self.embedding_dim}")

        return embeddings, labels

