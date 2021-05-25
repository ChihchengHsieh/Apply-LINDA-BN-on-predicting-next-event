from CustomExceptions.Exceptions import PermuatationException
from torch.jit import Error
from Models.BaseNNModel import BaseNNModel
from Models import BaselineLSTMModel_V2
from Utils.VocabDict import VocabDict
from datetime import datetime
from LINDA_BN import learn, permute

from Utils.PrintUtils import print_big
from Parameters.Enums import PermuatationSampleDist, SelectableDatasets, SelectableLoss, SelectableModels
import torch
from Parameters.PredictingParameters import PredictingParameters
import os
import json
from CustomExceptions import NotSupportedError
import torch.nn as nn
import sys
import numpy as np
import pandas as pd
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
import pydotplus as dot
from IPython.core.display import SVG
from Data import XESDataset

from Parameters import EnviromentParameters, TrainingParameters
import matplotlib.pyplot as plt


class ExplainingController_V2:
    ######################################
    #   Initialisation
    ######################################
    def __init__(self, parameters: TrainingParameters, predicting_parameters: PredictingParameters) -> None:
        self.parameters = parameters
        self.predicting_parameters = predicting_parameters

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.__initialise_data()
        self.__initialise_model()

        # Load trained model
        if not self.predicting_parameters.load_model_folder_path is None:
            self.load_trained_model(
                self.predicting_parameters.load_model_folder_path)
        else:
            raise Exception(
                "You need to specify the path to load the trained model")

        self.__initialise_loss_fn()

    def __initialise_model(
        self,
    ):
        # Setting up model
        if self.parameters.model == SelectableModels.BaseLineLSTMModel:
            self.model = BaselineLSTMModel_V2(
                device=self.device,
                vocab=self.vocab,
                embedding_dim=self.parameters.baselineLSTMModelParameters.embedding_dim,
                lstm_hidden=self.parameters.baselineLSTMModelParameters.lstm_hidden,
                dropout=self.parameters.baselineLSTMModelParameters.dropout,
                num_lstm_layers=self.parameters.baselineLSTMModelParameters.num_lstm_layers,
            )
        elif self.parameters.model == SelectableModels.BaseNNModel:
            self.model = BaseNNModel(
                feature_names=self.feature_names,
                hidden_dim=self.parameters.baseNNModelParams.hidden_dim,
                dropout=self.parameters.baseNNModelParams.dropout
            )
        else:
            raise NotSupportedError("Model you selected is not supported")
        self.model.to(self.device)

    def __initialise_data(self):

        # Load vocab dict
        dataset = self.parameters.dataset
        ############# Sequential dataset need to load vocab #############
        if dataset == SelectableDatasets.BPI2012:
            vocab_dict_path = os.path.join(
                EnviromentParameters.BPI2020Dataset.preprocessed_foldr_path,
                XESDataset.get_type_folder_name(
                    self.parameters.bpi2012.BPI2012_include_types),
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
        elif dataset == SelectableDatasets.Diabetes:
            self.feature_names = EnviromentParameters.DiabetesDataset.feature_names
            self.target_name = EnviromentParameters.DiabetesDataset.target_name
        elif dataset == SelectableDatasets.BreastCancer:
            self.feature_names = EnviromentParameters.BreastCancerDataset.feature_names
            self.target_name = EnviromentParameters.BreastCancerDataset.target_name
        else:
            raise NotSupportedError("Dataset you selected is not supported")

    def load_trained_model(self, folder_path: str):

        # Load model
        model_loading_path = os.path.join(
            folder_path, EnviromentParameters.model_save_file_name)
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
        self.model.eval()
        print_big("Model loaded successfully from %s" % (model_loading_path))

    def __initialise_loss_fn(self):
        # Setting up loss
        if self.parameters.loss == SelectableLoss.CrossEntropy:
            self.loss = nn.CrossEntropyLoss(
                reduction="mean",
                ignore_index=self.model.vocab.padding_index(),
            )
        elif self.parameters.loss == SelectableLoss.BCE:
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

        if not type(self.model) == BaselineLSTMModel_V2:
            raise NotSupportedError("Unsupported model")

        data_predicted_list: list[int] = self.model.predicting_from_list_of_vacab_trace(
            data=[data], n_steps=n_steps, use_argmax=use_argmax)[0]

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
            self.model.vocab.list_of_index_to_vocab(p) for p in predicted_list]

        col_names = ["step_%d" % (i+1) for i in range(len(data))] + \
            ["predict_%d" % (n+1) for n in range(n_steps)]

        df_to_dump = pd.DataFrame(predicted_vocab_list, columns=[col_names])

        # Save the predicted and prediction to path
        os.makedirs('./Permutations', exist_ok=True)
        file_path = './Permutations/%s_permuted.csv' % str(datetime.now())

        df_to_dump.to_csv(file_path, index=False)

        bn = learn.learnBN(
            file_path, algorithm=learn.BN_Algorithm.HillClimbing)

        infoBN = gnb.getInformation(
            bn, size=EnviromentParameters.default_graph_size)

        # compute Markov Blanket
        markov_blanket = gum.MarkovBlanket(bn, col_names[-1])
        markov_blanket_dot = dot.graph_from_dot_data(markov_blanket.toDot())
        markov_blanket_dot.set_bgcolor("transparent")
        markov_blanket_html = SVG(markov_blanket_dot.create_svg()).data

        inference = gnb.getInference(
            bn, evs={col_names[-1]: data_predicted_list[-1]}, targets=col_names, size=EnviromentParameters.default_graph_size)

        os.remove(file_path)
        return df_to_dump, data_predicted_list, bn, gnb.getBN(bn, size=EnviromentParameters.default_graph_size), inference, infoBN, markov_blanket_html

    def medical_check_boundary(self, input_data: torch.tensor, variance: float = 0.1, steps: int = 10):

        ###### Scale the input ######
        norm_data = self.model.normalize_input(input_data)

        ###### Get prediction ######
        predicted_value = self.model(norm_data)

        ### Generate permutations
        norm_data = norm_data.squeeze()
        all_permutations = permute.generate_fix_step_permutation_for_finding_boundary(norm_data, variance=variance, steps=steps)
        all_permutation_t = torch.stack(all_permutations)

        all_result = self.model(all_permutation_t)

        ### Split by features 
        all_permutation_chunks = torch.split(all_permutation_t, steps*2, dim=0)
        all_result_chunks = torch.split(all_result, steps*2, dim=0)

        ### Grouping by feature
        group_lists = []
        for i, col_name in enumerate(self.feature_names):
            group_lists.append({
                "name": col_name,
                "permutations" :all_permutation_chunks[i],
                "results": all_result_chunks[i],
                "index": i
            })

        fig, axs = plt.subplots(len(group_lists), figsize=(10, 17))
        for i in range(len(group_lists)):
            index_in_row = group_lists[i]["index"]
            all_f_dots =group_lists[i]["permutations"][:,index_in_row].tolist()
            all_f_results = group_lists[i]["results"].squeeze().tolist()
            ## Append input data
            input_data_value = norm_data[index_in_row].item() 
            all_f_dots.append(input_data_value)
            all_f_results.append(predicted_value)

            axs[i].hlines(0, min(all_f_dots), max(all_f_dots))
            axs[i].vlines(input_data_value, 1, -1)
            axs[i].set_xlim(min(all_f_dots)-variance, max(all_f_dots)+variance)
            axs[i].set_ylim(-0.5, 0.5)
            true_array = np.array([all_f_dots[idx] for idx, r in enumerate(all_f_results) if r > 0.5 ])
            false_array = np.array([all_f_dots[idx] for idx, r in enumerate(all_f_results) if r <= 0.5 ])
            axs[i].plot(true_array, np.zeros_like(true_array) , 'bo', ms=8, mfc='b', label="True")
            axs[i].plot(false_array, np.zeros_like(false_array), 'ro', ms=8, mfc='r', label="False")
            axs[i].set_title(group_lists[i]['name'])
            axs[i].legend()

        fig.tight_layout()


    def medical_predict_lindaBN_explain(self, data, num_samples, variance=0.5, number_of_bins=4, sample_dist: PermuatationSampleDist = PermuatationSampleDist.Uniform, using_qcut: bool = True, clip_permutation=True):
        if not type(self.model) == BaseNNModel:
            raise NotSupportedError("Unsupported model")

        ###### Scale the input ######
        norm_data = self.model.normalize_input(data)

        ###### Get prediction ######
        predicted_value = self.model(norm_data)

        #################### Generate permutations ####################

        if sample_dist == PermuatationSampleDist.Uniform:
            all_permutations_t = permute.generate_permutation_for_numerical_all_dim(
                norm_data.squeeze(), num_samples=num_samples, variance=variance, clip_permutation=clip_permutation)
        elif sample_dist == PermuatationSampleDist.Normal:
            all_permutations_t = permute.generate_permutations_for_normerical_all_dim_normal_dist(
                norm_data.squeeze(), num_samples=num_samples, variance=variance)
        else:
            raise NotSupportedError(
                "Doesn't support this sampling distribution for generating permutations.")

        ################## Predict permutations ##################
        all_predictions = self.model(all_permutations_t)

        ################## Descretise numerical ##################
        reversed_permutations_t = self.model.reverse_normalize_input(
            all_permutations_t)
        permutations_df = pd.DataFrame(
            reversed_permutations_t.tolist(), columns=self.feature_names)
        self.permutations_df = permutations_df
        q = np.array(range(number_of_bins+1))/(1.0*number_of_bins)
        cat_df_list = []
        for col in permutations_df.columns.values:
            if col != self.target_name:
                if using_qcut:
                    cat_df_list.append(pd.DataFrame(pd.qcut(
                        permutations_df[col], q, duplicates='drop', precision=2), columns=[col]))
                else:
                    cat_df_list.append(pd.DataFrame(pd.cut(
                        permutations_df[col], number_of_bins, duplicates='drop', precision=2), columns=[col]))
            else:
                cat_df_list.append(pd.DataFrame(
                    permutations_df[col].values, columns=[col]))

        cat_df = pd.concat(cat_df_list, join="outer", axis=1)

        ########### add predicted value ###########
        cat_df[self.target_name] = (all_predictions > 0.5).squeeze().tolist()

        # Save the predicted and prediction to path
        os.makedirs('./Permutations', exist_ok=True)
        file_path = './Permutations/%s_permuted.csv' % str(datetime.now())

        cat_df.to_csv(file_path, index=False)

        bn = learn.learnBN(
            file_path, algorithm=learn.BN_Algorithm.HillClimbing)

        # if not has_more_than_one_predicted:
        #     raise PermuatationException("All permutation predict same results. Please increase variance or number of samples")

        if len(bn.arcs()) < 1:
            # raise PermuatationException("No relationships found between columns. Please increase variance or number of samples")
            infoBN = ""
        else:
            infoBN = gnb.getInformation(
                bn, size=EnviromentParameters.default_graph_size)

        # compute Markov Blanket
        markov_blanket = gum.MarkovBlanket(bn, self.target_name)
        markov_blanket_dot = dot.graph_from_dot_data(markov_blanket.toDot())
        markov_blanket_dot.set_bgcolor("transparent")
        markov_blanket_html = SVG(markov_blanket_dot.create_svg()).data

        # inference = gnb.getInference(
        #     bn, evs={self.target_name: to_infer}, targets=cat_df.columns.values, size="70")

        has_more_than_one_predicted = len(
            cat_df[self.target_name].unique()) > 1
        if has_more_than_one_predicted:
            input_evs = {self.target_name: "True"}
        else:
            input_evs = {}

        inference = gnb.getInference(
            bn, evs=input_evs, targets=cat_df.columns.values, size=EnviromentParameters.default_graph_size)

        os.remove(file_path)
        return cat_df, predicted_value.item(), bn, gnb.getBN(bn, size=EnviromentParameters.default_graph_size), inference, infoBN, markov_blanket_html

    ############################
    #   Utils
    ############################

    def show_model_info(self):

        print_big("Model Structure")
        sys.stdout.write(str(self.model))

        print_big("Loaded model has {%d} parameters" %
                  (self.model.num_all_params()))

    def generate_html_page_from_graphs(self, input, predictedValue, bn, inference, infoBN, markov_blanket):
        outputstring: str = "<h1 style=\"text-align: center\">Model</h1>" \
                            + "<div style=\"text-align: center\">" + self.predicting_parameters.load_model_folder_path + "</div>"\
                            + "<h1 style=\"text-align: center\">Input</h1>" \
                            + "<div style=\"text-align: center\">" + input + "</div>"\
                            + "<h1 style=\"text-align: center\">Predicted</h1>" \
                            + "<div style=\"text-align: center\">" + predictedValue + "</div>"\
                            + "<h1 style=\"text-align: center\">BN</h1>" \
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
            path_to_explanation, '%s_%s_graphs_LINDA-BN.html' % (os.path.basename(os.path.normpath(self.predicting_parameters.load_model_folder_path)), datetime.now()))
        with open(save_path, 'w')as output_file:
            output_file.write(html_content)

        print_big("HTML page has been saved to: %s" % (save_path))
