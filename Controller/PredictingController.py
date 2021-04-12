import os
import torch
import pathlib

import numpy as np
import torch.nn as nn
import torch.optim as optim

from typing import Tuple
from datetime import datetime
from Utils.Constants import Constants
from torch.utils.data import DataLoader
from Data.BPI2012Dataset import BPI2012Dataset
from Models.BaselineLSMTModel import BaselineLSTMModel
from Controller.TrainingRecord import TrainingRecord
from CustomExceptions.Exceptions import NotSupportedError
from Parameters.PredictingParameters import PredictingParameters
from Parameters.Enums import SelectableDatasets, SelectableLoss, SelectableLrScheduler, SelectableModels, SelectableOptimizer


from Utils.PrintUtils import print_peforming_task


from Utils.PrintUtils import print_peforming_task


class PredictingController:
    def __init__(self) -> None:
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        # Load standard dataset
        if (PredictingParameters.standard_dataset == SelectableDatasets.BPI2012):
            self.dataset = BPI2012Dataset(filePath=PredictingParameters.bpi_2012_path,
                                          preprocessed_folder_path=PredictingParameters.preprocessed_bpi_2012_folder_path,
                                          preprocessed_df_type=PredictingParameters.preprocessed_df_type,
                                          )
        else:
            raise NotSupportedError("Dataset you selected is not supported")

        # Initialise dataloaders
        self.train_data_loader = DataLoader(
            self.dataset, batch_size=PredictingParameters.batch_size, shuffle=True, collate_fn=self.dataset.collate_fn,
            # num_workers=4,
            # worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed()) % (2**32-1)),
        )

        # Setting up model
        # From the folder load the parameters for that model
        
        if (PredictingParameters.model == SelectableModels.BaseLineLSTMModel):
            self.model = BaselineLSTMModel(
                vocab_size=self.dataset.vocab_size(),
                embedding_dim=TrainingParameters.BaselineLSTMModelParameters.embedding_dim,
                lstm_hidden=TrainingParameters.BaselineLSTMModelParameters.lstm_hidden,
                dropout=TrainingParameters.BaselineLSTMModelParameters.dropout,
                num_lstm_layers=TrainingParameters.BaselineLSTMModelParameters.num_lstm_layers,
                paddingValue=self.dataset.vocab_to_index(
                    Constants.PAD_VOCAB),
            )
        else:
            raise NotSupportedError("Model you selected is not supported")


