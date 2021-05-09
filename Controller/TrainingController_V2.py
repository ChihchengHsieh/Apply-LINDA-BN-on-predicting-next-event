import os
import sys
import json
import torch
import pathlib

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sn
import pandas as pd

from typing import Tuple
from datetime import datetime
from torch.utils.data import DataLoader
from Controller.TrainingRecord import TrainingRecord
from CustomExceptions.Exceptions import NotSupportedError
from Parameters.TrainingParameters import TrainingParameters
from Data import BPI2012Dataset_V2, DiabeteDataset
from Parameters.Enums import (
    SelectableDatasets,
    SelectableLoss,
    SelectableLrScheduler,
    SelectableModels,
    SelectableOptimizer
)

from Models import BaselineLSTMModel_V2, BaseNNModel

from Parameters import EnviromentParameters

from Utils.PrintUtils import (
    print_big,
    print_peforming_task,
    print_taks_done,
    replace_print_flush,
)

class TrainingController_V2:
    model_save_file_name = "model.pt"

    #########################################
    #   Initialisation
    #########################################

    def __init__(self):
        ############ determine the device ############
        self.device: str = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.plot_cm = TrainingParameters.plot_cm

        print_big("Running on %s " % (self.device))

        ############ Initialise records ############
        self.record = TrainingRecord(
            record_freq_in_step=TrainingParameters.run_validation_freq
        )

        ############ Initialise counters ############
        self.__epoch: int = 0
        self.__steps: int = 0
        self.test_accuracy: float = None
        self.stop_epoch = TrainingParameters.stop_epoch

        self.__intialise_dataset()

        ############ Load model and optimizer ############
        if not TrainingParameters.load_model_folder_path is None:
            ############ Load trained if specified ############
            self.load_trained_model(
                TrainingParameters.load_model_folder_path,
                TrainingParameters.load_optimizer,
            )
            if not TrainingParameters.load_optimizer:
                self.__intialise_optimizer()
        else:
            ############ Load empty model ############
            print_big("Model initialised")
            self.__initialise_model()
            self.__intialise_optimizer()

        ############ move model to device ############
        self.model.to(self.device)

        self.__initialise_loss_fn()

    def __intialise_dataset(self):
        ############ Determine dataset ############
        if TrainingParameters.dataset == SelectableDatasets.BPI2012:
            self.dataset = BPI2012Dataset_V2(
                device=self.device,
                file_path=EnviromentParameters.BPI2020Dataset.file_path,
                preprocessed_folder_path=EnviromentParameters.BPI2020Dataset.preprocessed_foldr_path,
                preprocessed_df_type=EnviromentParameters.BPI2020Dataset.preprocessed_df_type,
                include_types=TrainingParameters.BPI2012.BPI2012_include_types,
            )
        elif TrainingParameters.dataset == SelectableDatasets.Diabetes:
            self.dataset = DiabeteDataset(
                device=self.device,
                file_path= EnviromentParameters.DiabetesDataset.file_path
            )
        else:
            raise NotSupportedError("Dataset you selected is not supported")

        # Create datasets
        # Lengths for each set
        train_dataset_len = int(
            len(self.dataset) * TrainingParameters.train_test_split_portion[0]
        )
        test_dataset_len = int(
            len(self.dataset) * TrainingParameters.train_test_split_portion[-1]
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
                TrainingParameters.dataset_split_seed
            ),
        )

        # Initialise dataloaders
        self.train_data_loader = DataLoader(
            self.train_dataset,
            batch_size=TrainingParameters.batch_size,
            shuffle=self.train_dataset.dataset.get_train_shuffle(),
            collate_fn=self.dataset.collate_fn,
            sampler= self.train_dataset.dataset.get_sampler_from_df(self.train_dataset[:])
            # num_workers=4,
            # worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed()) % (2**32-1)),

        )
        self.validation_data_loader = DataLoader(
            self.validation_dataset,
            batch_size=TrainingParameters.batch_size,
            shuffle=True,
            collate_fn=self.dataset.collate_fn,
        )
        self.test_data_loader = DataLoader(
            self.test_dataset,
            batch_size=TrainingParameters.batch_size,
            shuffle=True,
            collate_fn=self.dataset.collate_fn,
        )

    def __initialise_model(
        self,
    ):
        # Setting up model
        if TrainingParameters.model == SelectableModels.BaseLineLSTMModel:
            self.model = BaselineLSTMModel_V2(
                vocab=self.dataset.vocab,
                embedding_dim=TrainingParameters.BaselineLSTMModelParameters.embedding_dim,
                lstm_hidden=TrainingParameters.BaselineLSTMModelParameters.lstm_hidden,
                dropout=TrainingParameters.BaselineLSTMModelParameters.dropout,
                num_lstm_layers=TrainingParameters.BaselineLSTMModelParameters.num_lstm_layers,
            )
        elif TrainingParameters.model == SelectableModels.BaseNNModel:
            self.model = BaseNNModel(
                num_input_features= self.dataset.num_features(),
                hidden_dim = TrainingParameters.BaseNNModelParams.hidden_dim,
                dropout = TrainingParameters.BaseNNModelParams.dropout
            )
        else:
            raise NotSupportedError("Model you selected is not supported")
        self.model.to(self.device)

    def __intialise_optimizer(
        self,
    ):
        # Setting up optimizer
        if TrainingParameters.optimizer == SelectableOptimizer.Adam:
            self.opt = optim.Adam(
                self.model.parameters(),
                lr=TrainingParameters.OptimizerParameters.learning_rate,
                weight_decay=TrainingParameters.OptimizerParameters.l2,
            )
        elif TrainingParameters.optimizer == SelectableOptimizer.SGD:
            self.opt = optim.SGD(
                self.model.parameters(),
                lr=TrainingParameters.OptimizerParameters.learning_rate,
                weight_decay=TrainingParameters.OptimizerParameters.l2,
                momentum=TrainingParameters.OptimizerParameters.SGD_momentum,
            )
        else:
            raise NotSupportedError("Optimizer you selected is not supported")

        # Setting up the learning rate scheduler
        if (
            TrainingParameters.OptimizerParameters.scheduler
            == SelectableLrScheduler.StepScheduler
        ):
            self.scheduler = optim.lr_scheduler.StepLR(
                self.opt,
                step_size=TrainingParameters.OptimizerParameters.lr_scheduler_step,
                gamma=TrainingParameters.OptimizerParameters.lr_scheduler_gamma,
            )
        elif (
            TrainingParameters.OptimizerParameters.scheduler
            == SelectableLrScheduler.NotUsing
        ):
            self.scheduler = None
        else:
            raise NotSupportedError(
                "Learning rate scheduler you selected is not supported"
            )

    def __initialise_loss_fn(self):
        # Setting up loss
        if TrainingParameters.loss == SelectableLoss.CrossEntropy:
            self.loss = nn.CrossEntropyLoss(
                reduction="mean",
                ignore_index=self.dataset.vocab.padding_index(),
            )
        elif TrainingParameters.loss == SelectableLoss.BCE:
            self.loss = nn.BCELoss(
                reduction="mean",
            )
        else:
            raise NotSupportedError(
                "Loss function you selected is not supported")

    ##########################################
    #   Train & Evaluation
    ##########################################

    def train(
        self,
    ):
        self.model.to(self.device)
        while self.__epoch < self.stop_epoch:
            for _, train_data in enumerate(
                self.train_data_loader
            ):
                _, train_loss, train_accuracy = self.train_step(
                    train_data
                )
                self.__steps += 1

                if self.__steps % TrainingParameters.verbose_freq == 0:
                    replace_print_flush(
                        "| Epoch [%d] | Step [%d] | lr [%.6f] | Loss: [%.4f] | Acc: [%.4f]|"
                        % (
                            self.__epoch,
                            self.__steps,
                            self.opt.param_groups[0]["lr"],
                            train_loss,
                            train_accuracy,
                        )
                    )

                if (
                    self.__steps > 0
                    and self.__steps % TrainingParameters.run_validation_freq == 0
                ):
                    print_peforming_task("Validation")
                    (
                        validation_loss,
                        validation_accuracy,
                    ) = self.perform_eval_on_dataloader(self.validation_data_loader, show_report=False)
                    self.record.record_training_info(
                        train_accuracy=train_accuracy,
                        train_loss=train_loss,
                        validation_accuracy=validation_accuracy,
                        validation_loss=validation_loss,
                    )
                    self.record.plot_records()

            self.__epoch += 1

        print_taks_done("Training")
        print_peforming_task("Testing")
        self.perform_eval_on_testset()

        # Peform testing in the end

    def train_step(
        self, data
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Return is a tuple of (loss, accuracy)
        """
        self.model.train()
        self.opt.zero_grad()
        out, loss, accuracy = self.step(data)
        self.data = data
        self.out = out
        self.loss_value = loss
        loss.backward()
        self.opt.step()
        if not self.scheduler is None:
            self.scheduler.step()
        
        return out, loss.item(), accuracy.item()

    def step(self, data):
        # Make sure the last item in data is target
        target = data[-1]
        out = self.model.data_forward(data)
        loss = self.model.get_loss(self.loss, out, target)
        accuracy = self.model.get_accuracy(out, target)
        return out, loss, accuracy

    def eval_step(
        self, data
    ):
        """
        Return is a tuple of (loss, accuracy)
        """
        self.model.eval()
        out, loss, accuracy = self.step(data)
        return out, loss.item(), accuracy.item()

    def perform_eval_on_testset(self):
        print_peforming_task("Testing")
        _, self.test_accuracy = self.perform_eval_on_dataloader(
            self.test_data_loader, show_report=True)

    def perform_eval_on_dataloader(self, dataloader: DataLoader, show_report:bool=False) -> Tuple[float, float]:
        all_loss = []
        all_accuracy = []
        all_batch_size = []
        all_predictions = []
        all_targets = []
        for _, data in enumerate(dataloader):
            target = data[-1]
            mask = self.model.generate_mask(target)
            out, loss, accuracy = self.eval_step(data)
            all_predictions.extend(self.model.get_prediction_list_from_out(out, mask ))
            all_targets.extend(self.model.get_target_list_from_target(target, mask))
            all_loss.append(loss)
            all_accuracy.append(accuracy)
            all_batch_size.append(len(data[-1]))

        accuracy = accuracy_score(all_targets, all_predictions)
        mean_loss = (torch.tensor(all_loss) * torch.tensor(all_batch_size)).sum() / len(
            dataloader.dataset
        )
        
        print_big(
            "Evaluation result | Loss [%.4f] | Accuracy [%.4f] "
            % (mean_loss, accuracy)
        )

        
        if (show_report):
            print_big("Classification Report")
            report = classification_report(all_targets, all_predictions, zero_division=0, output_dict=True, labels=list(range(len(self.model.get_labels()))),target_names=list(self.model.get_labels()))
            print(pd.DataFrame(report))

            print_big("Confusion Matrix")
            self.plot_confusion_matrix(all_targets, all_predictions)

        return mean_loss.item(), accuracy

    def plot_confusion_matrix(self, targets: list[int], predictions:  list[int]):
        # Plot the cufusion matrix
        cm = confusion_matrix(targets, predictions, labels=list(
            range(len(self.model.get_labels()))))
        df_cm = pd.DataFrame(cm, index=list(
            self.model.get_labels()), columns=list(self.model.get_labels()))

        if (self.plot_cm):
            plt.figure(figsize=(40, 40), dpi=100)
            sn.heatmap(df_cm / np.sum(cm), annot=True, fmt='.2%')
        else:
            print("="*20)
            print(df_cm)
            print("="*20)

    #######################################
    #   Utils
    #######################################
    def show_model_info(self):

        print_big("Model Structure")
        sys.stdout.write(str(self.model))

        print_big("Loaded model has {%d} parameters" %
                  (self.model.num_all_params()))

        if (self.__steps != 0):
            print_big(
                "Loaded model has been trained for [%d] steps, [%d] epochs"
                % (self.steps, self.epoch)
            )

            self.record.plot_records()

    def plot_grad_flow(self):
        """
        Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
        """
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in self.model.named_parameters():
            if (p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plt.bar(np.arange(len(max_grads)), max_grads,
                alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads,
                alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        # zoom in on the lower gradient regions
        plt.ylim(bottom=-0.001, top=0.02)
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend(
            [
                Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4),
            ],
            ["max-gradient", "mean-gradient", "zero-gradient"],
        )
        plt.show()

    #####################################
    #   Save
    #####################################
    def save_training_result(self, train_file: str) -> None:
        """
        Save to SavedModels folder:
        """
        if not (self.test_accuracy is None):
            saving_folder_path = os.path.join(
                pathlib.Path(train_file).parent,
                "SavedModels/%.4f_%s_%s_%s" % (self.test_accuracy, TrainingParameters.dataset.value, TrainingParameters.model.value,
                                         str(datetime.now())),
            )
        else:
             saving_folder_path = os.path.join(
                pathlib.Path(train_file).parent,
                "SavedModels/%s_%s_%s" % (TrainingParameters.dataset.value, TrainingParameters.model.value,
                                         str(datetime.now())),
            )

        # Create folder for saving
        os.makedirs(saving_folder_path, exist_ok=True)

        # Save parameters
        parameters_saving_path = os.path.join(
            saving_folder_path, TrainingParameters.parameters_save_file_name__
        )
        TrainingParameters.save_parameters_json__(parameters_saving_path)

        # Save training records
        records_saving_path = os.path.join(
            saving_folder_path, TrainingRecord.records_save_file_name
        )
        self.record.save_records_to_file(records_saving_path)

        # Save training figure
        figure_saving_path = os.path.join(
            saving_folder_path, TrainingRecord.figure_save_file_name
        )
        self.record.save_figure(figure_saving_path)

        # Save model
        model_saving_path = os.path.join(
            saving_folder_path, self.model_save_file_name
        )
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.opt.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "epoch": self.__epoch,
                "steps": self.__steps,
            },
            model_saving_path,
        )

        print_big("Model saved successfully to: %s " % (saving_folder_path))

    #########################################
    #   Load
    #########################################

    def resume_training(self, folder_path: str):
        '''
        Load a trained model to resume training.
        Use this function to load `Model`, `Dataset` and `Optimizer` to prevent unwanted result.
        [folder_path]: path to the trained model.
        '''

        self.training_parameters = self.load_training_parameters(
            folder_path=folder_path)
        self.intialise_dataset_with_parameters(self, self.training_parameters)
        self.load_trained_model(
            folder_path=folder_path, load_optimizer=TrainingParameters.load_optimizer, parameters=self.training_parameters)

    def load_training_parameters(self, folder_path: str):
        parameters_loading_path = os.path.join(
            folder_path, TrainingParameters.parameters_save_file_name__
        )
        with open(parameters_loading_path, "r") as output_file:
            parameters = json.load(output_file)
        return parameters

    def load_trained_model(self, folder_path: str, load_optimizer: bool, parameters= None):
        records_loading_path = os.path.join(
            folder_path, TrainingRecord.records_save_file_name
        )
        self.record.load_records(records_loading_path)

        # Load parameters (Parameters is needed to load model and optimzers)
        if parameters == None:
            parameters = self.load_training_parameters(folder_path=folder_path)

        # Create model according parameters
        self.__buid_model_with_parameters(parameters)

        # Load model
        model_loading_path = os.path.join(
            folder_path, self.model_save_file_name)
        checkpoint = torch.load(
            model_loading_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)

        if load_optimizer:
            # Create optimizer according to the parameters
            self.__build_optimizer_with_parameters(parameters)
            self.opt.load_state_dict(checkpoint["optimizer_state_dict"],)
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])


        self.__epoch = checkpoint["epoch"]
        self.__steps = checkpoint["steps"]

        del checkpoint

        print_big("Model loaded successfully from: %s " % (folder_path))


    ################################################
    # Build with parameters
    ################################################

    def intialise_dataset_with_parameters(self, parameters):
        '''
        When a training process is resumed, this method will be used to load
        exact same data (Prevent the training data to infect testing data).
        [parameters]: parameters containing all the training information.
        '''

        selectedDataset = SelectableDatasets[parameters["dataset"]]

        ########################################
        # Load dataset
        ########################################
        if selectedDataset == SelectableDatasets.BPI2012:
            self.dataset = BPI2012Dataset_V2(
                filePath=EnviromentParameters.BPI2020Dataset.bpi_2012_path,
                preprocessed_folder_path=EnviromentParameters.BPI2020Dataset.preprocessed_foldr_path,
                preprocessed_df_type=EnviromentParameters.BPI2020Dataset.preprocessed_df_type,
            )

        elif selectedDataset == SelectableDatasets.Diabetes:
            self.dataset = DiabeteDataset(
                file_path=EnviromentParameters.DiabetesDataset.file_path
            )
        else:
            raise NotSupportedError("Dataset you selected is not supported")

        ########################################
        # Determine the size for each dataset
        ########################################

        train_dataset_len = int(
            len(self.dataset) * parameters["train_test_split_portion"][0]
        )
        test_dataset_len = int(
            len(self.dataset) * parameters["train_test_split_portion"][-1]
        )
        validation_dataset_len = len(self.dataset) - (
            train_dataset_len + test_dataset_len
        )

        #######################################
        # Split dataset
        #######################################
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

        #######################################
        # Create dataloaders
        #######################################
        self.train_data_loader = DataLoader(
            self.train_dataset,
            batch_size=parameters["batch_size"],
            shuffle=self.train_dataset.dataset.get_train_shuffle(),
            collate_fn=self.dataset.collate_fn,
            sampler= self.train_dataset.dataset.get_sampler_from_df(self.train_dataset[:])

            # num_workers=4,
            # worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed()) % (2**32-1)),
        )
        self.validation_data_loader = DataLoader(
            self.validation_dataset,
            batch_size=parameters["batch_size"],
            shuffle=True,
            collate_fn=self.dataset.collate_fn,
        )
        self.test_data_loader = DataLoader(
            self.test_dataset,
            batch_size=parameters["batch_size"],
            shuffle=True,
            collate_fn=self.dataset.collate_fn,
        )


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
                vocab=self.dataset.vocab,
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
                num_input_features=self.dataset.num_features(),
                hidden_dim=  parameters["BaseNNModelParams"]["hidden_dim"],
                dropout = parameters["BaseNNModelParams"]["dropout"],
            )
        
        else:
            raise NotSupportedError("Model you selected is not supported")

    def __build_optimizer_with_parameters(self, parameters):
        '''
        The parameters usually can be found in the SavedModels folder.
        If a training want to resume after unloading, this method can be used
        to load recorded optimizer parameters.
        [parameters]: parameters containing all the training information.
        '''

        selectedOptimizer = SelectableOptimizer[parameters["optimizer"]] 

        ###########################
        # Set up optimizer
        ###########################
        if selectedOptimizer == SelectableOptimizer.Adam:
            self.opt = optim.Adam(
                self.model.parameters(),
                lr=parameters["OptimizerParameters"]["learning_rate"],
                weight_decay=parameters["OptimizerParameters"]["l2"],
            )
        elif selectedOptimizer == SelectableOptimizer.SGD:
            self.opt = optim.SGD(
                self.model.parameters(),
                lr=parameters["OptimizerParameters"]["learning_rate"],
                weight_decay=parameters["OptimizerParameters"]["l2"],
                momentum=parameters["OptimizerParameters"]["SGD_momentum"],
            )
        else:
            raise NotSupportedError("Optimizer you selected is not supported")

        ################################
        # Set up lr scheduler
        ################################
        shceduler= SelectableLrScheduler[parameters["OptimizerParameters"]["scheduler"]]

        if shceduler == SelectableLrScheduler.StepScheduler:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.opt,
                step_size=parameters["OptimizerParameters"]["lr_scheduler_step"],
                gamma=parameters["OptimizerParameters"]["lr_scheduler_gamma"],
            )
        elif shceduler == SelectableLrScheduler.NotUsing:
            self.scheduler = None
        else:
            raise NotSupportedError(
                "Learning rate scheduler you selected is not supported"
            )
