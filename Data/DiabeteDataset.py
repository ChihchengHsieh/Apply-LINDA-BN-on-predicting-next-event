import pandas as pd
from Utils.FileUtils import file_exists
from torch.utils.data import Dataset
import torch
import numpy as np


class DiabeteDataset(Dataset):
    def __init__(self, file_path: str, device: torch.device = torch.device("cpu")):
        self.device = device
        self.df = pd.read_csv(file_path)
        self.target_col_name = "Outcome"
        self.feature_names =[ col  for col in self.df.columns if col != self.target_col_name ]

    def __len__(self) -> int:
        return len(self.df)

    def num_features(self) -> int:
        return len(self.feature_names)

    def __getitem__(self, index: int) -> pd.Series:
        return self.df.iloc[index]

    def collate_fn(self, data: list[pd.Series]) -> tuple[torch.Tensor, torch.Tensor]:
        # Transform to df
        input_df = pd.DataFrame(data)
        input_data = input_df[self.feature_names]
        input_target = input_df[self.target_col_name]
        return torch.tensor(np.array(input_data)).to(self.device).float(), torch.tensor(np.array(input_target)).to(self.device).float()


   