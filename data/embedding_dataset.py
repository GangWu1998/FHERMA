import torch 
from torch.utils.data import Dataset, DataLoader 
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split

class EmbeddingDataSet(Dataset):
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray,
                normalize: bool = True):
        """
        Args:
            embeddings: pretrain embedding vectors [N, embedding_dim]
            labels: label [N]
            normalize: is the embedding normalize
        """
        self.embeddings = torch.tensor(embeddings, dtype = torch.float32)
        self.labels = torch.tensor(labels, dtype = torch.long)

        if normalize:
            #L2归一化
            self.embeddings = torch.nn.functional.normalize(self.embeddings, p = 2, dim = 1)
        
        def __len__(self) -> int:
            return len(self.embeddings)
        
        def __getitem__(self, idx: int) ->Tuple[torch.Tensor, torch.Tensor]:
            return self.embeddings[idx], self.labels[idx]
        
        def create_embedding_dataloaders(embeddings: np.ndarray, labels: np.ndarray,
                                        test_size: float = 0.2, batch_size: int = 32,
                                        normalize: bool = True) -> Tuple[DataLoader, DataLoader]:
            """create embedding data loader"""
            X_train, X_test, y_train, y_test = train_test_split(
                embeddings, labels, test_size, random_state = 42, stratify = labels
            )
            train_dataset = EmbeddingDataSet(X_train, y_train, normalize = normalize)
            test_dataset = EmbeddingDataSet(X_test, y_test, normalize = normalize)

            train_loader = DataLoader(train_dataset, batch_size = batch_size,
                                     shuffle = true, num_workers = 2)
            test_loader = DataLoader(test_dataset, batch_size = batch_size,
                                     shuffle = False, num_workers = 2)
            
            return train_loader, test_loader

        

