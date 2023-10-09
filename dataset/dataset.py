import numpy as np
import pandas as pd
import torch.utils.data as data

class NMTDataset(data.Dataset):
    def __init__(self, df):
        self.df = pd.read_csv(df)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return idx