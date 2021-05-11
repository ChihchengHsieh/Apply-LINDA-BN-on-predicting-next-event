import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from Utils import Constants

class VocabDict:
    def __init__(self, vocab_dict) -> None:
        self.vocab_dict = vocab_dict

    def index_to_vocab(self, index: int) -> str:
        for k, v in self.vocab_dict.items():
            if (v == index):
                return k
            continue

    def vocab_to_index(self, vocab: str) -> int:
        return self.vocab_dict[vocab]

    def list_of_index_to_vocab(self, list_of_index: list[int]):
        return [self.index_to_vocab(i) for i in list_of_index]

    def list_of_vocab_to_index(self, list_of_vocab: list[str]):
        return [self.vocab_to_index(v) for v in list_of_vocab]

    def vocab_size(self) -> int:
        '''
        Include <START>, <END> and <PAD> tokens. So, if you the actual number of activities,
        you have to minus 3.
        '''
        return len(self.vocab_dict)
    
    def padding_index(self):
        return self.vocab_to_index(Constants.PAD_VOCAB)

    def tranform_to_input_data_from_seq_idx_with_caseid(self, seq_list: list[list[int]], caseids: list[str] = None):
        '''
        Calculate the lengths for reach trace, so we can use padding.
        '''
        
        seq_lens = np.array([len(s)for s in seq_list])
        sorted_len_index = np.flip(np.argsort(seq_lens))
        sorted_seq_lens = [seq_lens[idx] for idx in sorted_len_index]
        sorted_seq_list = [torch.tensor(seq_list[idx])
                           for idx in sorted_len_index]

        if (caseids):
            sorted_caseids = [caseids[i] for i in sorted_len_index]
        else:
            sorted_caseids = None

        return sorted_caseids, pad_sequence(sorted_seq_list, batch_first=True, padding_value=0), torch.tensor(sorted_seq_lens)

    def __len__(self):
        return self.vocab_size()
