from datetime import datetime
from LINDA_BN import learn, permute
from typing import Tuple

from torch.nn.utils.rnn import pad_sequence
from Utils.Constants import Constants
from Models.BaselineLSMTModel import BaselineLSTMModel
from Utils.PrintUtils import print_big, print_peforming_task, print_taks_done
from Parameters.Enums import SelectableDatasets, SelectableLoss, SelectableModels
from Data.BPI2012Dataset import BPI2012Dataset
import torch
from Parameters.PredictingParameters import PredictingParameters
import os
import json
from CustomExceptions.Exceptions import NotSupportedError
import torch.nn as nn
import sys
import numpy as np
import pandas as pd
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb


class ExplainingController:
    def __init__(self) -> None:
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.training_parameters = self.load_training_parameters(
            PredictingParameters.load_model_folder_path
        )

        # Load vocab dict
        if self.training_parameters["dataset"] == str(SelectableDatasets.BPI2012):
            vocab_dict_path = os.path.join(
                PredictingParameters.preprocessed_bpi_2012_folder_path, BPI2012Dataset.vocab_dict_file_name)
            with open(vocab_dict_path, 'r') as output_file:
                self.vocab_dict = json.load(output_file)
        else:
            raise NotSupportedError("Dataset you selected is not supported")

        # Load trained model
        if not PredictingParameters.load_model_folder_path is None:
            self.load_trained_model(
                PredictingParameters.load_model_folder_path)
        else:
            raise Exception(
                "You need to specify the path to load the trained model")

        self.model.to(self.device)

        self.__initialise_loss_fn()

    def load_training_parameters(self, folder_path: str):
        parameters_loading_path = os.path.join(
            folder_path, PredictingParameters.parameters_save_file_name__
        )
        with open(parameters_loading_path, "r") as output_file:
            parameters = json.load(output_file)
        return parameters

    def load_trained_model(self, folder_path: str):

        # Create model according to the training parameters
        self.__buid_model_with_parameters(self.training_parameters)

        # Load model
        model_loading_path = os.path.join(
            folder_path, self.model.model_save_file_name)
        checkpoint = torch.load(
            model_loading_path, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint["model_state_dict"])

        print_big("Model loaded successfully")

    def __buid_model_with_parameters(self, parameters):
        # Setting up model
        if parameters["model"] == str(SelectableModels.BaseLineLSTMModel):
            self.model = BaselineLSTMModel(
                vocab_size=self.vocab_size(),
                embedding_dim=parameters["BaselineLSTMModelParameters"][
                    "embedding_dim"
                ],
                lstm_hidden=parameters["BaselineLSTMModelParameters"]["lstm_hidden"],
                dropout=parameters["BaselineLSTMModelParameters"]["dropout"],
                num_lstm_layers=parameters["BaselineLSTMModelParameters"][
                    "num_lstm_layers"
                ],
                paddingValue=self.vocab_to_index(Constants.PAD_VOCAB),
            )
        else:
            raise NotSupportedError("Model you selected is not supported")

    def __initialise_loss_fn(self):
        # Setting up loss
        if PredictingParameters.loss == SelectableLoss.CrossEntropy:
            self.loss = nn.CrossEntropyLoss()
        else:
            raise NotSupportedError(
                "Loss function you selected is not supported")

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

    def predict_next_n_for_trace(self, data: list[str],  n_steps: int = None, use_argmax=False) -> list[int]:
        return self.predicting_from_list_of_vacab_trace([data], n_steps=n_steps, use_argmax=use_argmax)[0]

    def predict_next_lindaBN_explain(self, data: list[str], n_steps=1, use_argmax=True):
        # Make prediction first
        data_predicted_list: list[int] = self.predict_next_n_for_trace(
            data=data, n_steps=n_steps, use_argmax=use_argmax)
        to_infer_vocab = data_predicted_list[-1]

        # Trnasfer to int list
        data_int_list = self.list_of_vocab_to_index(data)

        # generate permutations for input data
        all_permutations = permute.generate_permutation_for_trace(
            np.array(data_int_list), vocab_size=self.vocab_size())

        # Generate
        permutation_t = torch.tensor(all_permutations)
        predicted_list = self.predicting_from_list_of_idx_trace(
            data=permutation_t, n_steps=n_steps, use_argmax=use_argmax)

        # Convert to vocab list
        predicted_vocab_list = [
            self.list_of_index_to_vocab(p) for p in predicted_list]

        col_names = ["step_%d" % (i+1) for i in range(len(data))] + \
            ["predict_%d" % (n+1) for n in range(n_steps)]

        df_to_dump = pd.DataFrame(predicted_vocab_list, columns=[col_names])

        # Save the predicted and prediction to path
        os.makedirs('./Permutations', exist_ok=True)
        file_path = './Permutations/%s_permuted.csv' % str(datetime.now())

        df_to_dump.to_csv(file_path, index=False)

        bn, infoBN, essenceGraph = learn.learnBN(
            file_path, algorithm=learn.BN_Algorithm.HillClimbing)

        # compute Markov Blanket
        markov_blanket = gum.MarkovBlanket(bn, col_names[-1])

        inference = gnb.getInference(
            bn, evs={col_names[-1]: to_infer_vocab}, targets=col_names, size="70")

        os.remove(file_path)
        return data_predicted_list, gnb.getBN(bn), inference, infoBN, markov_blanket._repr_html_()

    def generate_html_page_from_graphs(self, bn, inference, infoBN, markov_blanket):
        outputstring: str = "<h1 style=\"text-align: center\">BN</h1>" \
                            + "<div style=\"text-align: center\">" + bn + "</div>"\
                            + ('</br>'*5) + "<h1 style=\"text-align: center\">Inference</h1>" \
                            + inference + ('</br>'*5) + "<h1 style=\"text-align: center\">Info BN</h1>"\
                            + infoBN + ('</br>'*5) + "<h1 style=\"text-align: center\">Markov Blanket</h1>"\
                            + "<div style=\"text-align: center\">" \
                            + markov_blanket + "</div>"
        return outputstring

    def predicting_from_list_of_vacab_trace(
        self, data: list[list[str]], n_steps: int = None, use_argmax=False
    ):
        print_peforming_task("Predicting")

        data = [self.list_of_vocab_to_index(l) for l in data]

        predicted_list = self.predicting_from_list_of_idx_trace(
            data=data, n_steps=n_steps, use_argmax=use_argmax
        )

        predicted_list = [
            self.list_of_index_to_vocab(l) for l in predicted_list
        ]

        print_taks_done("Predicting")

        return predicted_list

    def predicting_from_list_of_idx_trace(
        self, data: list[list[int]], n_steps: int = None, use_argmax=False
    ):
        data, lengths = self.tranform_to_input_data_from_seq_idx(data)

        predicted_list = self.predict(
            data=data, lengths=lengths, n_steps=n_steps, use_argmax=use_argmax
        )
        return predicted_list

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
                eos_idx=self.vocab_to_index(Constants.EOS_VOCAB),
                use_argmax=use_argmax,
                max_predicted_lengths=PredictingParameters.max_eos_predicted_length,
            )
        return predicted_list

    def padding_index(self):
        return self.vocab_to_index(Constants.PAD_VOCAB)

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
