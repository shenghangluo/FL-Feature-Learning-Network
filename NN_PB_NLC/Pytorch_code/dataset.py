import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, DataLoader


class DatasetDNN(Dataset):
    def __init__(self, X, Rx, Tx, transform=None):
        
        self.X = X
        self.Rx = Rx
        self.Tx = Tx
        self.transform = transform

    def __len__(self):
        return len(self.Rx)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        X = self.X[idx]
        Rx = self.Rx[idx]
        Tx = self.Tx[idx]


        sample = X,Rx,Tx

        if self.transform:
            sample = self.transform(sample)
        return sample

