from Utils.PrintUtils import print_big
from CustomExceptions.Exceptions import NotSupportedError
from json import load
from typing import Iterable, Union
import torch
import pandas as pd
from torch.jit import Error
from torch.utils.data import Dataset
import pm4py
from datetime import timedelta
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from Utils.Constants import Constants
from Utils.FileUtils import file_exists
import json
import os
from Parameters.Enums import PreprocessedDfType


class BPI2012Dataset(Dataset):
    pickle_df_file_name = "df.pickle"
    vocab_dict_file_name = "vocab_dict.json"

    def __init__(self, filePath: str, preprocessed_folder_path: str, preprocessed_df_type: PreprocessedDfType) -> None:
        super().__init__()

        self.filePath = filePath
        self.preprocessed_folder_path = preprocessed_folder_path
        self.preprocessed_df_type = preprocessed_df_type
        self.vocab_dict: dict(str, int)
        self.df: pd.DataFrame

        if (not preprocessed_folder_path is None) and self.preprocessed_data_exist(preprocessed_folder_path, preprocessed_df_type):
            self.load_preprocessed_data(
                preprocessed_folder_path,
                preprocessed_df_type,
            )
        else:
            self.__initialise_data(filePath=filePath)

            # Save preprocessed data.
            if not preprocessed_folder_path is None:
                self.save_preprocessed_data(
                    preprocessed_folder_path,
                    preprocessed_df_type,
                )

    def __initialise_data(self, filePath: str) -> None:
        log = pm4py.read_xes(filePath)
        flattern_log: list[dict[str, any]] = ([{**event,
                                                'caseid': trace.attributes['concept:name']}
                                               for trace in log for event in trace])
        df = pd.DataFrame(flattern_log)
        df["name_and_transition"] = df["concept:name"] + \
            "_" + df["lifecycle:transition"]
        df = df[['time:timestamp', 'name_and_transition', "caseid"]]
        newData = list()
        for case, group in df.groupby('caseid'):
            group.sort_values("time:timestamp", ascending=True, inplace=True)
            strating_time = group.iloc[0]["time:timestamp"] - \
                timedelta(microseconds=1)
            ending_time = group.iloc[-1]["time:timestamp"] + \
                timedelta(microseconds=1)
            traces = group.to_dict('record')

            # Add start and end tags.
            traces.insert(
                0, {"caseid": case, "time:timestamp": strating_time, "name_and_transition": Constants.SOS_VOCAB})
            traces.append(
                {"caseid": case, "time:timestamp": ending_time, "name_and_transition": Constants.EOS_VOCAB})
            newData.extend(traces)

        df = pd.DataFrame(newData)
        df['name_and_transition'] = df['name_and_transition'].astype(
            'category')

        vocab_dict: dict[str, int] = {}
        for i, cat in enumerate(df['name_and_transition'].cat.categories):
            # plus one, since we want to remain "0" for "<PAD>"
            vocab_dict[cat] = i+1
        vocab_dict[Constants.PAD_VOCAB] = 0

        # Create new index categorial column
        df['cat'] = df['name_and_transition'].apply(lambda c: vocab_dict[c])

        # Create the df only consist of trace and caseid
        final_df_data: list[dict[str, any]] = []
        for caseid, group in df.groupby('caseid'):
            final_df_data.append({
                "trace": list(group['cat']),
                "caseid": caseid
            })

        self.df: pd.DataFrame = pd.DataFrame(final_df_data)
        self.df.sort_values("caseid", inplace=True)
        self.vocab_dict: dict(str, int) = vocab_dict

    def longest_trace_len(self) -> int:
        return self.df.trace.map(len).max()

    def index_to_vocab(self, index: int) -> str:
        for k, v in self.vocab_dict.items():
            if (v == index):
                return k
            continue

    def list_of_index_to_vocab(self, list_of_index: list[int]):
        return [self.index_to_vocab(i) for i in list_of_index]

    def list_of_vocab_to_index(self, list_of_vocab: list[str]):
        return [self.vocab_to_index(v) for v in list_of_vocab]

    def vocab_to_index(self, vocab: str) -> int:
        return self.vocab_dict[vocab]

    def vocab_size(self) -> int:
        # This include tokens
        # minus 3 to remove <START>, <END> and <PAD>
        return len(self.vocab_dict)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> pd.Series:
        return self.df.iloc[index]

    @staticmethod
    def get_file_name_from_preprocessed_df_type(preprocessed_df_type: PreprocessedDfType):
        if preprocessed_df_type == PreprocessedDfType.Pickle:
            return BPI2012Dataset.pickle_df_file_name
        else:
            raise NotSupportedError(
                "Not supported saving format for preprocessed data")

    @staticmethod
    def preprocessed_data_exist(preprocessed_folder_path: str, preprocessed_df_type: PreprocessedDfType):
        file_name = BPI2012Dataset.get_file_name_from_preprocessed_df_type(
            preprocessed_df_type)
        df_path = os.path.join(preprocessed_folder_path, file_name)
        vocab_dict_path = os.path.join(
            preprocessed_folder_path, BPI2012Dataset.vocab_dict_file_name)
        return file_exists(df_path) and file_exists(vocab_dict_path)

    def store_df(self, preprocessed_folder_path: str, preprocessed_df_type: PreprocessedDfType):
        os.makedirs(preprocessed_folder_path, exist_ok=True)
        file_name = BPI2012Dataset.get_file_name_from_preprocessed_df_type(
            preprocessed_df_type)
        df_path = os.path.join(preprocessed_folder_path, file_name)
        if(preprocessed_df_type == PreprocessedDfType.Pickle):
            self.store_df_in_pickle(df_path)
        else:
            raise NotSupportedError(
                "Not supported saving format for preprocessed data")

    def load_df(self, preprocessed_folder_path: str, preprocessed_df_type: PreprocessedDfType):
        file_name = BPI2012Dataset.get_file_name_from_preprocessed_df_type(
            preprocessed_df_type)
        df_path = os.path.join(preprocessed_folder_path, file_name)
        if(preprocessed_df_type == PreprocessedDfType.Pickle):
            self.load_df_from_pickle(df_path)
        else:
            raise NotSupportedError(
                "Not supported loading format for preprocessed data")

    def store_df_in_pickle(self, path):
        self.df.to_pickle(path)

    def load_df_from_pickle(self, path):
        self.df = pd.read_pickle(path)

    def padding_index(self):
        return self.vocab_to_index(Constants.PAD_VOCAB)

    def save_preprocessed_data(self, preprocessed_folder_path: str, preprocessed_df_type: PreprocessedDfType):
        if preprocessed_folder_path is None:
            raise Error("Preprocessed folder path can't be None")

        # Store df
        self.store_df(preprocessed_folder_path, preprocessed_df_type)

        # Store vocab_dict
        vocab_dict_path = os.path.join(
            preprocessed_folder_path, BPI2012Dataset.vocab_dict_file_name)
        with open(vocab_dict_path, 'w') as output_file:
            json.dump(self.vocab_dict, output_file, indent='\t')

        print_big(
            "Preprocessed data saved successfully"
        )

    def load_preprocessed_data(self, preprocessed_folder_path: str, preprocessed_df_type: PreprocessedDfType):
        if preprocessed_folder_path is None:
            raise Error("Preprocessed folder path can't be None")

        # Load df
        self.load_df(preprocessed_folder_path, preprocessed_df_type)

        # load vocab_dict
        vocab_dict_path = os.path.join(
            preprocessed_folder_path, BPI2012Dataset.vocab_dict_file_name)
        with open(vocab_dict_path, 'r') as output_file:
            self.vocab_dict = json.load(output_file)

        print_big(
            "Preprocessed data loaded successfully"
        )

    @staticmethod
    def tranform_to_input_data_from_seq_idx_with_caseid(caseids: list[str], seq_list: list[list[int]]):
        seq_lens = np.array([len(s)for s in seq_list])
        sorted_len_index = np.flip(np.argsort(seq_lens))
        sorted_seq_lens = [seq_lens[idx] for idx in sorted_len_index]
        sorted_seq_list = [torch.tensor(seq_list[idx])
                           for idx in sorted_len_index]
        sorted_caseids = [caseids[i] for i in sorted_len_index]

        return sorted_caseids, pad_sequence(sorted_seq_list, batch_first=True, padding_value=0), torch.tensor(sorted_seq_lens)

    @staticmethod
    def tranform_to_input_data_from_seq_idx(seq_list: list[list[int]]):
        seq_lens = np.array([len(s)for s in seq_list])
        sorted_len_index = np.flip(np.argsort(seq_lens))
        sorted_seq_lens = [seq_lens[idx] for idx in sorted_len_index]
        sorted_seq_list = [torch.tensor(seq_list[idx])
                           for idx in sorted_len_index]

        return pad_sequence(sorted_seq_list, batch_first=True, padding_value=0), torch.tensor(sorted_seq_lens)

    @staticmethod
    def collate_fn(data: list[pd.Series]) -> Iterable[Union[np.ndarray, torch.Tensor, torch.Tensor, np.ndarray]]:
        caseid_list, seq_list = zip(
            *[(d["caseid"], torch.tensor(d["trace"])) for d in data])
        caseid_list = list(caseid_list)
        seq_list = list(seq_list)

        # Get sorting index
        seq_lens_before_splitting = np.array([len(s)for s in seq_list])
        sorted_len_index = np.flip(np.argsort(seq_lens_before_splitting))

        # Sort caseids and traces
        sorted_seq_list = [torch.tensor(seq_list[idx])
                           for idx in sorted_len_index]
        sorted_case_id = np.array(caseid_list)[sorted_len_index]

        # Build training and test seq
        # it should remove all the EOS to form a training set
        data_seq_list = [li[:-1] for li in sorted_seq_list]
        # it should remove all the SOS to form a testing set
        target_seq_list = [li[1:] for li in sorted_seq_list]

        # Get lengths
        data_seq_length = [len(l) for l in data_seq_list]

        # Pad data and target
        padded_data = pad_sequence(
            data_seq_list, batch_first=True,  padding_value=0)
        padded_target = pad_sequence(
            target_seq_list, batch_first=True, padding_value=0)

        return sorted_case_id, padded_data, padded_target, torch.tensor(data_seq_length)
