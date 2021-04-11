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
import pickle
import os
from Parameters.TrainingParameters import PreprocessedDfType


class BPI2012Dataset(Dataset):
    pickle_df_file_name = "df.pickle"
    hdf5_df_file_name = "store.h5"
    vocab_dict_file_name = "vocab_dict.pickle"

    def __init__(self, filePath: str, preprocessed_folder_path: str, preprocessed_df_type: PreprocessedDfType) -> None:
        super().__init__()

        self.filePath = filePath
        self.preprocessed_folder_path = preprocessed_folder_path
        self.preprocessed_df_type = preprocessed_df_type
        self.__vocab_dict: dict(str, int)
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
        flattern_log: list(dict(str, any)) = ([{**event,
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
        self.__vocab_dict: dict(str, int) = vocab_dict

    def longest_trace_len(self) -> int:
        return self.df.trace.map(len).max()

    def index_to_vocab(self, index: int) -> str:
        for k, v in self.__vocab_dict.items():
            if (v == index):
                return k
            continue

    def vocab_to_index(self, vocab: str) -> int:
        return self.__vocab_dict[vocab]

    def vocab_size(self) -> int:
        # This include tokens
        # minus 3 to remove <START>, <END> and <PAD>
        return len(self.__vocab_dict)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> pd.Series:
        return self.df.iloc[index]

    @staticmethod
    def get_file_name_from_preprocessed_df_type(preprocessed_df_type: PreprocessedDfType):
        if preprocessed_df_type == PreprocessedDfType.HDF5:
            return BPI2012Dataset.hdf5_df_file_name
        elif preprocessed_df_type == PreprocessedDfType.Pickle:
            return BPI2012Dataset.pickle_df_file_name
        else:
            raise NotSupportedError("Not supported saving format for preprocessed data")

    @staticmethod
    def preprocessed_data_exist(preprocessed_folder_path: str, preprocessed_df_type: PreprocessedDfType):
        file_name = BPI2012Dataset.get_file_name_from_preprocessed_df_type(
            preprocessed_df_type)
        df_path = os.path.join(preprocessed_folder_path, file_name)
        vocab_dict_path = os.path.join(
            preprocessed_folder_path, BPI2012Dataset.vocab_dict_file_name)
        return file_exists(df_path) and file_exists(vocab_dict_path)

    def store_df(self, preprocessed_folder_path: str, preprocessed_df_type: PreprocessedDfType):
        os.makedirs(preprocessed_folder_path, exist_ok= True)
        file_name = BPI2012Dataset.get_file_name_from_preprocessed_df_type(
            preprocessed_df_type)
        df_path = os.path.join(preprocessed_folder_path, file_name)
        if (preprocessed_df_type == PreprocessedDfType.HDF5):
            self.store_df_in_hdf5(df_path)
        elif(preprocessed_df_type == PreprocessedDfType.Pickle):
            self.store_df_in_pickle(df_path)
        else:
            raise NotSupportedError(
                "Not supported saving format for preprocessed data")

    def load_df(self, preprocessed_folder_path: str, preprocessed_df_type: PreprocessedDfType):
        file_name = BPI2012Dataset.get_file_name_from_preprocessed_df_type(
            preprocessed_df_type)
        df_path = os.path.join(preprocessed_folder_path, file_name)
        if (preprocessed_df_type == PreprocessedDfType.HDF5):
            self.load_df_from_hdf5(df_path)
        elif(preprocessed_df_type == PreprocessedDfType.Pickle):
            self.load_df_from_pickle(df_path)
        else:
            raise NotSupportedError(
                "Not supported loading format for preprocessed data")

    def store_df_in_hdf5(self, path):
        store = pd.HDFStore(path)
        store['df'] = self.df

    def store_df_in_pickle(self, path):
        self.df.to_pickle(path)

    def load_df_from_hdf5(self, path):
        store = pd.HDFStore(path)
        self.df = store['df']

    def load_df_from_pickle(self, path):
        self.df = pd.read_pickle(path)

    def save_preprocessed_data(self, preprocessed_folder_path: str, preprocessed_df_type: PreprocessedDfType):
        if preprocessed_folder_path is None:
            raise Error("Preprocessed folder path can't be None")

        # Store df
        self.store_df(preprocessed_folder_path, preprocessed_df_type)

        # Store vocab_dict
        vocab_dict_path = os.path.join(
            preprocessed_folder_path, BPI2012Dataset.vocab_dict_file_name)
        with open(vocab_dict_path, 'wb') as output_file:
            pickle.dump(self.__vocab_dict, output_file,
                        protocol=pickle.HIGHEST_PROTOCOL)

        print(
            "========================================"+"\n" +
            "| Preprocessed data saved successfully |" + "\n" +
            "========================================"
        )

    def load_preprocessed_data(self, preprocessed_folder_path: str, preprocessed_df_type: PreprocessedDfType):
        if preprocessed_folder_path is None:
            raise Error("Preprocessed folder path can't be None")

        # Load df
        self.load_df(preprocessed_folder_path, preprocessed_df_type)

        # load vocab_dict
        vocab_dict_path = os.path.join(
            preprocessed_folder_path, BPI2012Dataset.vocab_dict_file_name)
        with open(vocab_dict_path, 'rb') as output_file:
            self.__vocab_dict = pickle.load(output_file)

        print(
            "========================================="+"\n" +
            "| Preprocessed data loaded successfully |" + "\n" +
            "========================================="
        )

    @staticmethod
    def collate_fn(data: list[pd.Series]) -> Iterable[Union[np.ndarray, torch.Tensor, torch.Tensor, np.ndarray]]:
        caseid_list, seq_list = zip(
            *[(d["caseid"], torch.tensor(d["trace"])) for d in data])
        caseid_list = list(caseid_list)
        seq_list = list(seq_list)

        se_lens = np.array([len(s)for s in seq_list])
        sorted_len_index = np.flip(np.argsort(se_lens))
        sorted_seq_lens = se_lens[sorted_len_index]
        sorted_seq_list = [seq_list[idx] for idx in sorted_len_index]
        sorted_case_id = np.array(caseid_list)[sorted_len_index]
        seq_tensor = pad_sequence(sorted_seq_list, batch_first=True)

        # since we reduce the length for train and target
        return sorted_case_id, seq_tensor[:, :-1], seq_tensor[:, 1:], torch.tensor(sorted_seq_lens - 1)
