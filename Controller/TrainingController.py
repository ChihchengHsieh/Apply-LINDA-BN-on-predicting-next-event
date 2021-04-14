from Data.PredictingJsonDataset import PredictingJsonDataset
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

from typing import Tuple
from datetime import datetime
from Utils.Constants import Constants
from torch.utils.data import DataLoader
from Data.BPI2012Dataset import BPI2012Dataset
from Models.BaselineLSMTModel import BaselineLSTMModel
from Controller.TrainingRecord import TrainingRecord
from CustomExceptions.Exceptions import NotSupportedError
from Parameters.TrainingParameters import TrainingParameters
from Parameters.Enums import (
    SelectableDatasets,
    SelectableLoss,
    SelectableLrScheduler,
    SelectableModels,
    SelectableOptimizer
)

from Utils.PrintUtils import (
    print_big,
    print_peforming_task,
    print_percentages,
    print_taks_done,
    replace_print_flush,
)


class TrainingController:
    def __init__(self):
        # determine the device
        self.device: str = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        print_big("Running on %s " % (self.device))

        # Initialise records
        self.record = TrainingRecord(
            record_freq_in_step=TrainingParameters.run_validation_freq
        )

        # Initialise counter
        self.__epoch: int = 0
        self.__steps: int = 0
        self.test_accuracy: float = None
        self.stop_epoch = TrainingParameters.stop_epoch

        self.__intialise_dataset()

        # Load trained model if need
        if not TrainingParameters.load_model_folder_path is None:
            self.load_trained_model(
                TrainingParameters.load_model_folder_path,
                TrainingParameters.load_optimizer,
            )
            if not TrainingParameters.load_optimizer:
                self.__intialise_optimizer()
        else:
            self.__initialise_model()
            self.__intialise_optimizer()

        self.model.to(self.device)

        self.__initialise_loss_fn()

    def __intialise_dataset(self):
        # Load dataset
        if TrainingParameters.dataset == SelectableDatasets.BPI2012:
            self.dataset = BPI2012Dataset(
                filePath=TrainingParameters.bpi_2012_path,
                preprocessed_folder_path=TrainingParameters.preprocessed_bpi_2012_folder_path,
                preprocessed_df_type=TrainingParameters.preprocessed_df_type,
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
            shuffle=True,
            collate_fn=self.dataset.collate_fn,
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
            self.model = BaselineLSTMModel(
                vocab_size=self.dataset.vocab_size(),
                embedding_dim=TrainingParameters.BaselineLSTMModelParameters.embedding_dim,
                lstm_hidden=TrainingParameters.BaselineLSTMModelParameters.lstm_hidden,
                dropout=TrainingParameters.BaselineLSTMModelParameters.dropout,
                num_lstm_layers=TrainingParameters.BaselineLSTMModelParameters.num_lstm_layers,
                paddingValue=self.dataset.vocab_to_index(Constants.PAD_VOCAB),
            )
        else:
            raise NotSupportedError("Model you selected is not supported")

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
                size_average=True,
                reduction="mean",
                ignore_index=self.dataset.padding_index(),
            )
        else:
            raise NotSupportedError(
                "Loss function you selected is not supported")

    def train(
        self,
    ):
        self.model.to(self.device)
        while self.__epoch < self.stop_epoch:
            for _, (_, train_data, train_target, train_lengths) in enumerate(
                self.train_data_loader
            ):
                train_data, train_target, train_lengths = (
                    train_data.to(self.device),
                    train_target.to(self.device),
                    train_lengths.to(self.device),
                )
                train_loss, train_accuracy = self.train_step(
                    train_data, train_target, train_lengths
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
                    ) = self.perform_eval_on_dataloader(self.validation_data_loader)
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
        self, train_data: torch.tensor, target: torch.tensor, lengths: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Return is a tuple of (loss, accuracy)
        """
        self.model.train()
        self.opt.zero_grad()
        loss, accuracy = self.model_step(train_data, target, lengths)
        loss.backward()
        self.opt.step()
        if not self.scheduler is None:
            self.scheduler.step()

        return loss.item(), accuracy.item()

    def eval_step(
        self, validation_data: torch.tensor, target: torch.tensor, lengths: torch.tensor
    ):
        """
        Return is a tuple of (loss, accuracy)
        """
        self.model.eval()
        loss, accuracy = self.model_step(validation_data, target, lengths)
        return loss, accuracy

    def model_step(
        self, input: torch.tensor, target: torch.tensor, lengths: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Return is a tuple of (loss, accuracy)
        """
        out, _ = self.model(input, lengths)

        loss = self.loss(out.transpose(2, 1), target)

        # Mask the padding

        accuracy = torch.mean(
            torch.masked_select(
                (torch.argmax(out, dim=-1) == target), target > 0
            ).float()
        )

        return loss, accuracy

    def perform_eval_on_testset(self):
        print_peforming_task("Testing")
        _, self.test_accuracy = self.perform_eval_on_dataloader(
            self.test_data_loader)

    def perform_eval_on_dataloader(self, dataloader: DataLoader) -> Tuple[float, float]:
        all_loss = []
        all_accuracy = []
        all_batch_size = []
        for _, (_, data, target, lengths) in enumerate(dataloader):
            data, target, lengths = (
                data.to(self.device),
                target.to(self.device),
                lengths.to(self.device),
            )
            loss, accuracy = self.eval_step(data, target, lengths)
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
        return mean_loss.item(), mean_accuracy.item()

    def save_training_result(self, train_file: str) -> None:
        """
        Save to SavedModels folder:
        """
        if not (self.test_accuracy is None):
            saving_folder_path = os.path.join(
                pathlib.Path(train_file).parent,
                "SavedModels/%.4f_%s" % (self.test_accuracy,
                                         str(datetime.now())),
            )
        else:
            saving_folder_path = os.path.join(
                pathlib.Path(train_file).parent,
                "SavedModels/{}".format(str(datetime.now())),
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
            saving_folder_path, self.model.model_save_file_name
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

    def load_training_parameters(self, folder_path: str):
        parameters_loading_path = os.path.join(
            folder_path, TrainingParameters.parameters_save_file_name__
        )
        with open(parameters_loading_path, "r") as output_file:
            parameters = json.load(output_file)
        return parameters

    def load_trained_model(self, folder_path: str, load_optimizer: bool, parameters):
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
            folder_path, self.model.model_save_file_name)
        checkpoint = torch.load(model_loading_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if load_optimizer:
            # Create optimizer according to the parameters
            self.__build_optimizer_with_parameters(parameters)
            self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.__epoch = checkpoint["epoch"]
        self.__steps = checkpoint["steps"]

        print_big("Model loaded successfully from: %s " % (folder_path))

    def __buid_model_with_parameters(self, parameters):
        # Setting up model
        if parameters["model"] == str(SelectableModels.BaseLineLSTMModel):
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

    def __build_optimizer_with_parameters(self, parameters):
        # Setting up optimizer
        if parameters["optimizer"] == str(SelectableOptimizer.Adam):
            self.opt = optim.Adam(
                self.model.parameters(),
                lr=parameters["OptimizerParameters"]["learning_rate"],
                weight_decay=parameters["OptimizerParameters"]["l2"],
            )
        elif parameters["optimizer"] == str(SelectableOptimizer.SGD):
            self.opt = optim.SGD(
                self.model.parameters(),
                lr=parameters["OptimizerParameters"]["learning_rate"],
                weight_decay=parameters["OptimizerParameters"]["l2"],
                momentum=parameters["OptimizerParameters"]["SGD_momentum"],
            )
        else:
            raise NotSupportedError("Optimizer you selected is not supported")

        # Setting up the learning rate scheduler
        if parameters["OptimizerParameters"]["scheduler"] == str(
            SelectableLrScheduler.StepScheduler
        ):
            self.scheduler = optim.lr_scheduler.StepLR(
                self.opt,
                step_size=parameters["OptimizerParameters"]["lr_scheduler_step"],
                gamma=parameters["OptimizerParameters"]["lr_scheduler_gamma"],
            )
        elif parameters["OptimizerParameters"]["scheduler"] == str(
            SelectableLrScheduler.NotUsing
        ):
            self.scheduler = None
        else:
            raise NotSupportedError(
                "Learning rate scheduler you selected is not supported"
            )

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

    def perform_eval_on_dataset(self):
        print_peforming_task("Evaluation")
        temp_dataloader = DataLoader(
            self.dataset,
            batch_size=TrainingParameters.batch_size,
            shuffle=True,
            collate_fn=self.dataset.collate_fn,
            # num_workers=4,
            # worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed()) % (2**32-1)),
        )
        self.perform_eval_on_dataloader(temp_dataloader)

    def load_json_for_predicting(
        self, path: str, n_steps: int = None, use_argmax=False
    ):

        print_peforming_task("Dataset Loading")
        # Expect it to be a 2D list contain list of traces.
        p_dataset = PredictingJsonDataset(
            self.dataset, predicting_file_path=path)

        p_loader = DataLoader(
            p_dataset,
            batch_size=TrainingParameters.batch_size,
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
        self, data: list[list[str]], n_steps: int = None, use_argmax=False
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
                max_predicted_lengths=TrainingParameters.max_eos_predicted_length,
            )
        return predicted_list

    def plot_grad_flow(self):
        """Plots the gradients flowing through different layers in the net during training.
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

    def resume_training(self, folder_path: str):
        training_parameters = self.load_training_parameters(
            folder_path=folder_path)
        self.intialise_dataset_with_parameters(self, training_parameters)
        self.load_trained_model ( folder_path= folder_path,load_optimizer= TrainingParameters.load_optimizer,parameters= training_parameters  )
        self.training_parameters = training_parameters

    def intialise_dataset_with_parameters(self, parameters):
        # Load standard dataset
        if parameters["dataset"] == str(SelectableDatasets.BPI2012):
            self.dataset = BPI2012Dataset(
                filePath=TrainingParameters.bpi_2012_path,
                preprocessed_folder_path=TrainingParameters.preprocessed_bpi_2012_folder_path,
                preprocessed_df_type=TrainingParameters.preprocessed_df_type,
            )
        else:
            raise NotSupportedError("Dataset you selected is not supported")

        # Initialise dataloaders
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=TrainingParameters.batch_size,
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
            batch_size=parameters["batch_size"],
            shuffle=True,
            collate_fn=self.dataset.collate_fn,
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
