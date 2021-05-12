from Utils.VocabDict import VocabDict
import torch
from torch.utils.data import Dataset
import json


class PredictingJsonDataset(Dataset):

    def __init__(self, vocab: VocabDict, predicting_file_path: str ,device: torch.device = torch.device("cpu")) -> None:
        '''
        Expect file structure:
        {
            "__caseid__": [ vocab_#1, vocab_#2 ]
        }
        '''
        super().__init__()
        self.device = device
        self.vocab = vocab
        self.predicting_file_path = predicting_file_path

        with open(self.predicting_file_path, "r") as file:
            retrieved_dict: dict[str, list[str]] = json.load(file)

        self.data = [(k, v) for k, v in retrieved_dict.items()]

    def __getitem__(self, index: int) -> tuple[str, list[str]]:
        d = self.data[index]
        return d[0], d[1]

    def __len__(self) -> int:
        return len(self.data)

    def collate_fn(self, data: list[tuple[str, list[str]]]):
        caseids, seq = zip(*data)

        seq = [self.vocab.list_of_vocab_to_index(l) for l in seq]

        return self.vocab.tranform_to_input_data_from_seq_idx_with_caseid(
            caseids, seq)

        # return sorted_caseids, pad_sequence(sorted_seq_list, batch_first=True, padding_value=0), torch.tensor(sorted_seq_lens)
