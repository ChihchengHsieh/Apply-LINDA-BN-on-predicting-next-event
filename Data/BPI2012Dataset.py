from typing import Tuple
from pandas.core.frame import DataFrame
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pm4py
from datetime import timedelta
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence
import torch.nn.functional as F


class BPI2012Dataset(Dataset):
    def __init__(self, filePath: str) -> None:
        super().__init__()
        self.__initialise_data(filePath=filePath)
        self.df: pd.DataFrame
        self.__event_dict: dict(str, int)

    def __initialise_data(self, filePath: str) -> None:
        log = pm4py.read_xes(filePath)
        flattern_log: list(dict(str, any)) = ([{**event,
                                                'caseid': trace.attributes['concept:name']}
                                               for trace in log for event in trace])
        df = pd.DataFrame(flattern_log)
        df["name_and_transition"] = df["concept:name"] + \
            "_" + df["lifecycle:transition:"]
        df = df[['time:timestamp', 'name_and_transition']]
        newData = list()
        for case, group in df.groupby('caseid'):
            group.sort_values("time:timestamp", ascending=True, inplace=True)
            strating_time = group.iloc[0]["time:timestamp"] - \
                timedelta(microseconds=1)
            ending_time = group.iloc[-1]["time:timestamp"] + \
                timedelta(microseconds=1)
            traces = group.to_dict('record')
            traces.insert(
                0, {"caseid": case, "time:timestamp": strating_time, "name_and_transition": "<START>"})
            traces.append(
                {"caseid": case, "time:timestamp": ending_time, "name_and_transition": "<END>"})
            newData.extend(traces)

        df = pd.DataFrame(newData)
        df['name_and_transition'] = df['name_and_transition'].astype(
            'category')

        event_dict = dict(str, int)
        for i, cat in enumerate(df['name_and_transition'].cat.categories):
            # plus one, since we want to remain "0" for "<PAD>"
            event_dict[cat] = i+1
        event_dict["<PAD>"] = 0

        # Create new index categorial column
        df['cat'] = df['name_and_transition'].apply(lambda c: event_dict[c])

        # Create the df only consist of trace and caseid
        final_df_data = list(dict(str, any))
        for caseid, group in df.groupby('caseid'):
            final_df_data.append({
                "trace": list(group['cat']),
                "caseid": caseid
            })

        self.df: pd.DataFrame = pd.DataFrame(final_df_data)
        self.__event_dict: dict(str, int) = event_dict

    def longest_trace_len(self) -> int:
        return self.df.trace.map(len).max()

    def index_to_event(self, index: int) -> str:
        for k, v in self.__event_dict.items():
            if (v == index):
                return k
            continue

    def event_to_index(self, event: str) -> int:
        return self.__event_dict[event]

    def num_of_unique_events(self) -> int:
        # This include tokens
        # minus 3 to remove <START>, <END> and <PAD>
        return len(self.__event_dict)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> pd.Series:
        return self.df.iloc[index]

    def collate_fn(self, data: list(pd.Series)) -> Tuple(np.ndarray, torch.Tensor, torch.Tensor, np.ndarray):
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
        train_seq = F.one_hot(
            seq_tensor[:, :-1], num_classes=self.num_of_unique_events)
        target_seq = F.one_hot(
            seq_tensor[:, 1:], num_classes=self.num_of_unique_events)

        return sorted_case_id, train_seq, target_seq, sorted_seq_lens
