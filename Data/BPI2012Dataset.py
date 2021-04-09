import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pm4py


class BPI2012Dataset(Dataset):
    def __init__(self, filePath: str) -> None:
        super().__init__()
        log = pm4py.read_xes(filePath)
        
