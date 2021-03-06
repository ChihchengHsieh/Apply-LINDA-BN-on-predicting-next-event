import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler


class MedicalDataset(Dataset):
    def __init__(self, file_path: str, device: torch.device, target_col_name: str, feature_names: list[str]):
        self.device = device
        self.df = pd.read_csv(file_path)
        self.target_col_name = target_col_name
        if (feature_names is None):
            self.feature_names =[ col  for col in self.df.columns if col != self.target_col_name ]
        else:
            self.feature_names = feature_names

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

    def get_sampler_from_df(self, df, seed: int):
        target = df[self.target_col_name]
        class_sample_count = np.array(
          [len(np.where(target == t)[0]) for t in np.unique(target)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in target])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weigth = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), generator=torch.Generator().manual_seed(
                seed))
        return sampler

    def get_train_shuffle(self):
        return False
