from torch.jit import Error
import Models
from Models.BaseNNModel import BaseNNModel
from Models import BaselineLSTMModel_V2
from Utils.VocabDict import VocabDict
from datetime import datetime
from LINDA_BN import learn, permute
from typing import Tuple

from torch.nn.utils.rnn import pad_sequence
from Utils.Constants import Constants
from Models.BaselineLSMTModel import BaselineLSTMModel
from Utils.PrintUtils import print_big, print_peforming_task, print_taks_done
from Parameters.Enums import ActivityType, SelectableDatasets, SelectableLoss, SelectableModels
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
import pydotplus as dot
from IPython.core.display import SVG
from torch.utils.data import DataLoader
from Data import BPI2012Dataset_V2, XESDataset

from Parameters import EnviromentParameters


class ExplainingController:
    ######################################
    #   Initialisation
    ######################################

    def __init__(self) -> None:
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.training_parameters = self.load_training_parameters(
            PredictingParameters.load_model_folder_path
        )

        self.__initialise_data()
        
        # Load trained model
        if not PredictingParameters.load_model_folder_path is None:
            self.load_trained_model(
                PredictingParameters.load_model_folder_path)
        else:
            raise Exception(
                "You need to specify the path to load the trained model")

        self.__initialise_loss_fn()

    def __initialise_data(self):
        # Load vocab dict
        dataset = SelectableDatasets[self.training_parameters["dataset"]]
        ############# Sequential dataset need to load vocab #############
        if dataset == SelectableDatasets.BPI2012:
            vocab_dict_path = os.path.join(
                EnviromentParameters.BPI2020Dataset.preprocessed_foldr_path,
                XESDataset.get_type_folder_name([ActivityType[t]
                                                     for t in self.training_parameters["BPI2012_include_types"]]),
                XESDataset.vocab_dict_file_name)
            with open(vocab_dict_path, 'r') as output_file:
                vocab_dict = json.load(output_file)
                self.vocab = VocabDict(vocab_dict)
        elif dataset == SelectableDatasets.Helpdesk: 
            vocab_dict_path = os.path.join(
                EnviromentParameters.HelpDeskDataset.preprocessed_foldr_path,
                XESDataset.get_type_folder_name(),
                XESDataset.vocab_dict_file_name)
            with open(vocab_dict_path, 'r') as output_file:
                vocab_dict = json.load(output_file)
                self.vocab = VocabDict(vocab_dict)
        elif dataset  == SelectableDatasets.Diabetes:
            self.feature_names =  EnviromentParameters.DiabetesDataset.feature_names
        elif dataset == SelectableDatasets.BreastCancer:
            self.feature_names = EnviromentParameters.BreastCancerDataset.feature_names
        else:
            raise NotSupportedError("Dataset you selected is not supported")


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
        # TODO:
        # Mean and vriance will be calculated by standard scaler, but mean and variance will be store in the model.
        # The data will only be normalized when it's in the model?.
        # create data_forward for input normal data, and forward for input normalised data
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if (self.model.should_load_mean_and_vairance()):
            self.model.mean_ = checkpoint["mean_"]
            self.model.var_ = checkpoint["var_"]

        self.model.to(self.device)

        print_big("Model loaded successfully")

    def __buid_model_with_parameters(self, parameters):
        '''
        The parameters usually can be found in the SavedModels folder.
        If a training want to resume after unloading, this method can be used
        to load trained weight.
        [parameters]: parameters containing all the training information.
        '''
        selectedModel: SelectableModels = SelectableModels[parameters["model"]]

        ##########################
        # Build models
        ##########################
        if selectedModel == SelectableModels.BaseLineLSTMModel:
            self.model = BaselineLSTMModel_V2(
                vocab=self.vocab,
                embedding_dim=parameters["BaselineLSTMModelParameters"][
                    "embedding_dim"
                ],
                lstm_hidden=parameters["BaselineLSTMModelParameters"]["lstm_hidden"],
                dropout=parameters["BaselineLSTMModelParameters"]["dropout"],
                num_lstm_layers=parameters["BaselineLSTMModelParameters"][
                    "num_lstm_layers"
                ],
            )

        elif selectedModel == SelectableModels.BaseNNModel:
            self.model = BaseNNModel(
                feature_names= self.feature_names,
                hidden_dim=  parameters["BaseNNModelParams"]["hidden_dim"],
                dropout = parameters["BaseNNModelParams"]["dropout"],
            )
        
        else:
            raise NotSupportedError("Model you selected is not supported")

    def __initialise_loss_fn(self):
        # Setting up loss
        if PredictingParameters.loss == SelectableLoss.CrossEntropy:
            self.loss = nn.CrossEntropyLoss(
                reduction="mean",
                ignore_index=self.dataset.vocab.padding_index(),
            )
        elif PredictingParameters.loss == SelectableLoss.BCE:
            self.loss = nn.BCELoss(
                reduction="mean",
            )
        else:
            raise NotSupportedError(
                "Loss function you selected is not supported")

    #################################
    #   Explaination
    #################################
    def pm_predict_lindaBN_explain(self, data: list[str], n_steps=1, use_argmax=True):

        if not self.model is BaselineLSTMModel_V2:
            raise NotSupportedError("Unsupported model")

        data_predicted_list: list[int] = self.model.predicting_from_list_of_vacab_trace(
            data=[data], n_steps=n_steps, use_argmax=use_argmax)[0]
        to_infer_vocab = data_predicted_list[-1]

        # Trnasfer to int list
        data_int_list = self.model.vocab.list_of_vocab_to_index(data)

        # generate permutations for input data
        all_permutations = permute.generate_permutation_for_trace(
            np.array(data_int_list), vocab_size=self.model.vocab.vocab_size())

        # Generate
        permutation_t = torch.tensor(all_permutations)
        predicted_list = self.model.predicting_from_list_of_idx_trace(
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
        markov_blanket_dot = dot.graph_from_dot_data(markov_blanket.toDot())
        markov_blanket_dot.set_bgcolor("transparent")
        markov_blanket_html = SVG(markov_blanket_dot.create_svg()).data

        inference = gnb.getInference(
            bn, evs={col_names[-1]: to_infer_vocab}, targets=col_names, size="70")

        os.remove(file_path)
        return data_predicted_list, gnb.getBN(bn), inference, infoBN, markov_blanket_html
    
    def medical_predict_lindaBN_explain(self, data):
        if not self.model is BaseNNModel:
            raise NotSupportedError("Unsupported model")

        ###### Scale the input ######
        norm_data = self.model.normalize_input(data)

        ###### Get prediction ######         
        predicted_value = self.model(norm_data)


        ##################### Make prediction first #####################
        predicted_value = self.model.data_forward(data)

        


        pass

    ############################
    #   Utils
    ############################

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
    def generate_html_page_from_graphs(self, bn, inference, infoBN, markov_blanket):
        outputstring: str = "<h1 style=\"text-align: center\">BN</h1>" \
                            + "<div style=\"text-align: center\">" + bn + "</div>"\
                            + ('</br>'*5) + "<h1 style=\"text-align: center\">Inference</h1>" \
                            + inference + ('</br>'*5) + "<h1 style=\"text-align: center\">Info BN</h1>"\
                            + infoBN + ('</br>'*5) + "<h1 style=\"text-align: center\">Markov Blanket</h1>"\
                            + "<div style=\"text-align: center\">" \
                            + markov_blanket + "</div>"
        return outputstring

    def save_html(self, html_content: str):
        path_to_explanation = './Explanations'
        os.makedirs(path_to_explanation, exist_ok=True)
        save_path = os.path.join(
            path_to_explanation, '%s_graphs_LINDA-BN.html' % (datetime.now()))
        with open(save_path, 'w')as output_file:
            output_file.write(html_content)

        print_big("HTML page has been saved to: %s" % (save_path))