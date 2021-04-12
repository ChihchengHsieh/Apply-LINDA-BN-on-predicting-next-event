from CustomExceptions.Exceptions import NotSupportedError
from json import load
from typing import Iterable, Tuple, Union
from pandas.core.frame import DataFrame
import torch
import pandas as pd
from torch.jit import Error
from torch.utils.data import Dataset, DataLoader
import pm4py
from datetime import timedelta
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence
import torch.nn.functional as F
from Utils.Constants import Constants
from Utils.FileUtils import file_exists
import json
import os
from Parameters.Enums import PreprocessedDfType


class PredictingJsonDataset(Dataset):

    def __init__(self, standard_dataset: Dataset, predicting_file_path: str) -> None:
        '''
        Expect file structure:
        {
            "__caseid__": [ vocab_#1, vocab_#2 ]
        }
        '''
        super().__init__()
        self.standard_dataset = standard_dataset
        self.predicting_file_path = predicting_file_path

        with open(self.predicting_file_path, "r") as file:
            retrieved_dict: dict[str, list[str]] = json.load(file)

        self.data = [(k, v) for k, v in retrieved_dict.items()]

    def __getitem__(self, index: int) -> Tuple[str, list[str]]:
        d = self.data[index]
        return d[0], d[1]

    def __len__(self) -> int:
        return len(self.data)

    def collate_fn(self, data: list[Tuple[str, list[str]]]):
        caseids, seq = zip(*data)

        seq = [self.standard_dataset.list_of_vocab_to_index(l) for l in seq]

        return self.standard_dataset.tranform_to_input_data_from_seq_idx_with_caseid(
            caseids, seq)

        # return sorted_caseids, pad_sequence(sorted_seq_list, batch_first=True, padding_value=0), torch.tensor(sorted_seq_lens)
