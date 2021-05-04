from numpy.lib.function_base import select
from Data.PredictingJsonDataset import PredictingJsonDataset
import os
import json
import torch
import pathlib
import sys

import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from typing import Tuple
from Utils.Constants import Constants
from torch.utils.data import DataLoader
from Data.BPI2012Dataset import BPI2012Dataset
from Models.BaselineLSMTModel import BaselineLSTMModel
from Controller.TrainingRecord import TrainingRecord
from CustomExceptions.Exceptions import NotSupportedError
from Parameters.PredictingParameters import PredictingParameters
from Parameters.Enums import ActivityType, SelectableDatasets, SelectableLoss, SelectableModels

from Utils.PrintUtils import (
    print_big,
    print_peforming_task,
    print_percentages,
    print_taks_done
)


class PredictingController:
    def __init__(self) -> None:
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.record = TrainingRecord(record_freq_in_step=0)

        self.training_parameters = self.load_training_parameters(
            PredictingParameters.load_model_folder_path
        )

        self.__intialise_dataset_with_parameters(self.training_parameters)

        # Load trained model
        if not PredictingParameters.load_model_folder_path is None:
            self.load_trained_model(
                PredictingParameters.load_model_folder_path)
        else:
            raise Exception(
                "You need to specify the path to load the trained model")

        self.model.to(self.device)

        self.__initialise_loss_fn()

    def __initialise_loss_fn(self):
        # Setting up loss
        if PredictingParameters.loss == SelectableLoss.CrossEntropy:
            self.loss = nn.CrossEntropyLoss()
        else:
            raise NotSupportedError(
                "Loss function you selected is not supported")

    def __intialise_dataset_with_parameters(self, parameters):
        # Load standard dataset

        if SelectableDatasets[parameters["dataset"]] == SelectableDatasets.BPI2012:
            self.dataset = BPI2012Dataset(
                filePath=PredictingParameters.bpi_2012_path,
                preprocessed_folder_path=PredictingParameters.preprocessed_bpi_2012_folder_path,
                preprocessed_df_type=PredictingParameters.preprocessed_df_type,
                include_types=[ActivityType[t]
                               for t in parameters["BPI2012_include_types"]]
            )
        else:
            raise NotSupportedError("Dataset you selected is not supported")

        # Initialise dataloaders
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=PredictingParameters.batch_size,
            shuffle=True,
            collate_fn=self.dataset.collate_fn,
            # num_workers=4,
            # worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed()) % (2**32-1)),
        )

        # Create datasets
        # Lengths for each set
        train_dataset_len = int(
            len(self.dataset) * parameters["train_test_split_portion"][0]
        )
        test_dataset_len = int(
            len(self.dataset) * parameters["train_test_split_portion"][-1]
        )
        validation_dataset_len = len(self.dataset) - (
            train_dataset_len + test_dataset_len
        )

        # Split the dataset
        (
            self.train_dataset,
            self.validation_dataset,
            self.test_dataset,
        ) = torch.utils.data.random_split(
            dataset=self.dataset,
            lengths=[train_dataset_len,
                     validation_dataset_len, test_dataset_len],
            generator=torch.Generator().manual_seed(
                parameters["dataset_split_seed"]),
        )

        # Initialise dataloaders
        self.train_data_loader = DataLoader(
            self.train_dataset,
            batch_size=PredictingParameters.batch_size,
            shuffle=True,
            collate_fn=self.dataset.collate_fn,
            # num_workers=4,
            # worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed()) % (2**32-1)),
        )
        self.validation_data_loader = DataLoader(
            self.validation_dataset,
            batch_size=PredictingParameters.batch_size,
            shuffle=True,
            collate_fn=self.dataset.collate_fn,
        )
        self.test_data_loader = DataLoader(
            self.test_dataset,
            batch_size=PredictingParameters.batch_size,
            shuffle=True,
            collate_fn=self.dataset.collate_fn,
        )

    def load_training_parameters(self, folder_path: str):
        parameters_loading_path = os.path.join(
            folder_path, PredictingParameters.parameters_save_file_name__
        )
        with open(parameters_loading_path, "r") as output_file:
            parameters = json.load(output_file)
        return parameters

    def load_trained_model(self, folder_path: str):
        records_loading_path = os.path.join(
            folder_path, TrainingRecord.records_save_file_name
        )
        self.record.load_records(records_loading_path)

        # Create model according to the training parameters
        self.__buid_model_with_parameters(self.training_parameters)

        # Load model
        model_loading_path = os.path.join(
            folder_path, self.model.model_save_file_name)
        checkpoint = torch.load(
            model_loading_path, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.epoch = checkpoint["epoch"]
        self.steps = checkpoint["steps"]

        print_big("Model loaded successfully")

    def show_model_info(self):

        print_big("Model Structure")
        sys.stdout.write(str(self.model))

        print_big("Loaded model has {%d} parameters" %
                  (self.model.num_all_params()))

        print_big(
            "Loaded model has been trained for [%d] steps, [%d] epochs"
            % (self.steps, self.epoch)
        )

        self.record.plot_records()

    def model_step(
        self, input: torch.tensor, target: torch.tensor, lengths: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """
        Return is a tuple of (loss, accuracy)
        input: (B, S)
        target: (B, S)
        """
        out, _ = self.model(input, lengths)  # (B, S, Vocab)

        loss = self.loss(out.transpose(2, 1), target)

        # Get all the valid predictions

        mask = target > 0
        predicted = torch.argmax(out, dim=-1)  # (B, S)
        selected_predictions = torch.masked_select(
            predicted, mask)

        selected_targets = torch.masked_select(
            target, mask
        )

        accuracy = torch.mean(
            (selected_predictions == selected_targets).float())

        # # Get all the targets
        # accuracy = torch.mean(
        #     torch.masked_select(
        #         (torch.argmax(out, dim=-1) == target), target > 0
        #     ).float()
        # )

        return selected_predictions.tolist(), selected_targets.tolist(), loss, accuracy

    def eval_step(
        self, validation_data: torch.tensor, target: torch.tensor, lengths: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """
        Return is a tuple of (loss, accuracy)
        """
        self.model.eval()
        prediction_list, target_list, loss, accuracy = self.model_step(
            validation_data, target, lengths)

        # We can get all the ground truth and the predictions

        return prediction_list, target_list, loss, accuracy

    def perform_eval_on_whole_dataset(self):
        print_peforming_task("Evaluation on whole dataset")
        self.perform_eval_on_dataloader(self.dataloader)

    def perform_eval_on_testing_set(self):
        print_peforming_task("Evaluation on test dataset")
        self.perform_eval_on_dataloader(self.test_data_loader)

    def perform_eval_on_training_set(self):
        print_peforming_task("Evaluation on training dataset")
        self.perform_eval_on_dataloader(self.test_data_loader)

    def perform_eval_on_validation_set(self):
        print_peforming_task("Evaluation on validation dataset")
        self.perform_eval_on_dataloader(self.test_data_loader)

    def perform_eval_on_dataloader(self, dataloader: DataLoader) -> Tuple[float, float]:
        all_loss = []
        all_accuracy = []
        all_batch_size = []
        all_predictions = []
        all_targets = []
        for _, (_, data, target, lengths) in enumerate(dataloader):
            data, target, lengths = (
                data.to(self.device),
                target.to(self.device),
                lengths.to(self.device),
            )
            prediction_list, target_list, loss, accuracy = self.eval_step(
                data, target, lengths)
            all_predictions.extend(prediction_list)
            all_targets.extend(target_list)
            all_loss.append(loss)
            all_accuracy.append(accuracy)
            all_batch_size.append(len(lengths))

        mean_accuracy = (
            torch.tensor(all_accuracy) * torch.tensor(all_batch_size)
        ).sum() / len(dataloader.dataset)
        mean_loss = (torch.tensor(all_loss) * torch.tensor(all_batch_size)).sum() / len(
            dataloader.dataset
        )

        print_big(
            "Evaluation result | Accuracy [%.4f] | Loss [%.4f]"
            % (mean_accuracy, mean_loss)
        )

        self.all_predictions = all_predictions
        self.all_targets = all_targets

        print_big("Classification Report")
        print(classification_report(all_targets, all_predictions))

        print_big("Confusion Matrix")
        self.plot_confusion_matrix(all_targets, all_predictions)

        return mean_loss.item(), mean_accuracy.item()

    def plot_confusion_matrix(self, targets: list[int], predictions:  list[int]):
        # Plot the cufusion matrix
        cm = confusion_matrix(targets, predictions, labels=list(
            range(self.dataset.vocab_size())))
        self.cm = cm
        df_cm = pd.DataFrame(cm, index=list(
            self.dataset.vocab_dict.keys()), columns=list(self.dataset.vocab_dict.keys()))
        self.df_cm = df_cm
        plt.figure(figsize=(40, 40), dpi=100)
        sn.heatmap(df_cm / np.sum(cm), annot=True, fmt='.2%')

    def __buid_model_with_parameters(self, parameters):
        model = SelectableModels[parameters["model"]]

        # Setting up model
        if model == SelectableModels.BaseLineLSTMModel:
            self.model = BaselineLSTMModel(
                vocab_size=self.dataset.vocab_size(),
                embedding_dim=parameters["BaselineLSTMModelParameters"][
                    "embedding_dim"
                ],
                lstm_hidden=parameters["BaselineLSTMModelParameters"]["lstm_hidden"],
                dropout=parameters["BaselineLSTMModelParameters"]["dropout"],
                num_lstm_layers=parameters["BaselineLSTMModelParameters"][
                    "num_lstm_layers"
                ],
                paddingValue=self.dataset.vocab_to_index(Constants.PAD_VOCAB),
            )
        else:
            raise NotSupportedError("Model you selected is not supported")

    def load_json_for_predicting(
        self, path: str, n_steps: int = None, use_argmax=False
    ):

        print_peforming_task("Dataset Loading")
        # Expect it to be a 2D list contain list of traces.
        p_dataset = PredictingJsonDataset(
            self.dataset, predicting_file_path=path)

        p_loader = DataLoader(
            p_dataset,
            batch_size=PredictingParameters.batch_size,
            shuffle=False,
            collate_fn=p_dataset.collate_fn,
            # num_workers=4,
            # worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed()) % (2**32-1)),
        )

        print_taks_done("Dataset Loading")

        predicted_dict = {}

        print_peforming_task("Predicting")

        for _, (caseids, data, lengths) in enumerate(p_loader):

            data, lengths = data.to(self.device), lengths.to(self.device)

            predicted_list = self.predict(
                data=data, lengths=lengths, n_steps=n_steps, use_argmax=use_argmax
            )

            for k, v in zip(caseids, predicted_list):
                predicted_dict[k] = self.dataset.list_of_index_to_vocab(v)

            sys.stdout.write("\r")
            print_percentages(
                prefix="Predicting cases",
                percentage=len(predicted_dict) / len(p_dataset),
            )
            sys.stdout.flush()

        print_taks_done("Predicting")

        # Save as result json
        saving_path = pathlib.Path(path)
        saving_file_name = pathlib.Path(path).stem + "_result.json"
        saving_dest = os.path.join(saving_path.parent, saving_file_name)
        with open(saving_dest, "w") as output_file:
            json.dump(predicted_dict, output_file, indent="\t")

        print_big("Predition result has been save to: %s" % (saving_dest))

    def predicting_from_list_of_vacab_trace(
        self, data: list[list[str]], n_steps: int = None, use_argmax=False
    ):

        print_peforming_task("Predicting")

        data = [self.dataset.list_of_vocab_to_index(l) for l in data]

        predicted_list = self.predicting_from_list_of_idx_trace(
            data=data, n_steps=n_steps, use_argmax=use_argmax
        )

        predicted_list = [
            self.dataset.list_of_index_to_vocab(l) for l in predicted_list
        ]

        print_taks_done("Predicting")

        return predicted_list

    def predicting_from_list_of_idx_trace(
        self, data: list[list[int]], n_steps: int = None, use_argmax=False
    ):
        data, lengths = self.dataset.tranform_to_input_data_from_seq_idx(data)

        predicted_list = self.predict(
            data=data, lengths=lengths, n_steps=n_steps, use_argmax=use_argmax
        )
        return predicted_list

    def predict(
        self,
        data: torch.tensor,
        lengths: torch.tensor = None,
        n_steps: int = None,
        use_argmax=False,
    ):
        if not n_steps is None:
            # Predict for n steps
            predicted_list = self.model.predict_next_n(
                input=data, lengths=lengths, n=n_steps, use_argmax=use_argmax
            )
        else:
            # Predict till EOS
            predicted_list = self.model.predict_next_till_eos(
                input=data,
                lengths=lengths,
                eos_idx=self.dataset.vocab_to_index(Constants.EOS_VOCAB),
                use_argmax=use_argmax,
                max_predicted_lengths=PredictingParameters.max_eos_predicted_length,
            )
        return predicted_list
