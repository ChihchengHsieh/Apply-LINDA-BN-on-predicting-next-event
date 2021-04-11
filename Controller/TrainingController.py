import os
import torch
import pathlib

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
from Parameters.TrainingParameters import TrainingParameters
from Parameters.Enums import SelectableDatasets, SelectableLoss, SelectableLrScheduler, SelectableModels, SelectableOptimizer


from Utils.PrintUtils import print_peforming_task


class TrainingController:
    def __init__(self):
        # determine the device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Load dataset
        if (TrainingParameters.dataset == SelectableDatasets.BPI2012):
            self.dataset = BPI2012Dataset(filePath=TrainingParameters.bpi_2012_path,
                                          preprocessed_folder_path=TrainingParameters.preprocessed_bpi_2012_folder_path,
                                          preprocessed_df_type=TrainingParameters.preprocessed_df_type,
                                        )
        else:
            raise NotSupportedError("Dataset you selected is not supported")

        # Create datasets

        # Lengths for each set
        train_dataset_len = int(
            len(self.dataset) * TrainingParameters.train_test_split_portion[0])
        test_dataset_len = int(len(self.dataset) *
                               TrainingParameters.train_test_split_portion[-1])
        validation_dataset_len = len(
            self.dataset) - (train_dataset_len + test_dataset_len)

        # Split the dataset
        self.train_dataset, self.validation_dataset, self.test_dataset = torch.utils.data.random_split(
            dataset=self.dataset, lengths=[train_dataset_len, validation_dataset_len, test_dataset_len])

        # Initialise dataloaders
        self.train_data_loader = DataLoader(
            self.train_dataset, batch_size=TrainingParameters.batch_size, shuffle=True, collate_fn=self.dataset.collate_fn,)
        self.validation_data_loader = DataLoader(
            self.validation_dataset, batch_size=TrainingParameters.batch_size, shuffle=True, collate_fn=self.dataset.collate_fn)
        self.test_data_loader = DataLoader(
            self.test_dataset, batch_size=TrainingParameters.batch_size, shuffle=True, collate_fn=self.dataset.collate_fn)

        # Setting up model
        if (TrainingParameters.model == SelectableModels.BaseLineLSTMModel):
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

        # Setting up optimizer
        if TrainingParameters.optimizer == SelectableOptimizer.Adam:
            self.opt = optim.Adam(
                self.model.parameters(),
                lr=TrainingParameters.OptimizerParameters.learning_rate,
                weight_decay=TrainingParameters.OptimizerParameters.l2
            )
        else:
            raise NotSupportedError("Optimizer you selected is not supported")

        # Setting up the learning rate scheduler
        if (TrainingParameters.OptimizerParameters.scheduler == SelectableLrScheduler.StepScheduler):
            self.scheduler = optim.lr_scheduler.StepLR(
                self.opt,
                step_size=TrainingParameters.OptimizerParameters.lr_scheduler_step,
                gamma=TrainingParameters.OptimizerParameters.lr_scheduler_gamma,
            )
        elif TrainingParameters.OptimizerParameters.scheduler == SelectableLrScheduler.NotUsing:
            self.scheduler = None
        else:
            raise NotSupportedError(
                "Learning rate scheduler you selected is not supported")

        # Setting up loss
        if (TrainingParameters.loss == SelectableLoss.CrossEntropy):
            self.loss = nn.CrossEntropyLoss()
        else:
            raise NotSupportedError(
                "Loss function you selected is not supported")

        # Initialise counter
        self.epoch = 0
        self.steps = 0
        self.stop_epoch = TrainingParameters.stop_epoch

        # Initialise records
        self.record = TrainingRecord(
            record_freq_in_step=TrainingParameters.run_validation_freq)

        # Load trained model if need
        if not TrainingParameters.load_model_folder_path is None:
            self.load_trained_model(TrainingParameters.load_model_folder_path, TrainingParameters.load_optimizer)

    def train(self,):
        self.model.to(self.device)
        while self.epoch < self.stop_epoch:
            for _, (_, train_data, train_target, train_lengths) in enumerate(self.train_data_loader):
                train_data, train_target, train_lengths = train_data.to(
                    self.device), train_target.to(self.device), train_lengths.to(self.device)
                train_loss, train_accuracy = self.train_step(
                    train_data, train_target, train_lengths)
                self.steps += 1

                if self.steps % TrainingParameters.verbose_freq == 0:
                    print('| Epoch [%d] | Step [%d] | lr [%.6f] | Loss: [%.4f] | Acc: [%.4f]|' % (
                        self.epoch, self.steps, self.opt.param_groups[0]['lr'], train_loss, train_accuracy))

                if self.steps > 0 and self.steps % TrainingParameters.run_validation_freq == 0:
                    print_peforming_task("Validation")
                    validation_loss, validation_accuracy = self.perform_eval_on_dataloader(
                        self.validation_data_loader)
                    self.record.record_training_info(
                        train_accuracy=train_accuracy,
                        train_loss=train_loss,
                        validation_accuracy=validation_accuracy,
                        validation_loss=validation_loss
                    )
                    self.record.plot_records()

            self.epoch += 1

        print_peforming_task("Testing")
        self.perform_eval_on_dataloader(self.test_data_loader)

        # Peform testing in the end

    def train_step(self, train_data: torch.tensor, target: torch.tensor, lengths: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        '''
        Return is a tuple of (loss, accuracy)
        '''
        self.model.train()
        self.opt.zero_grad()
        loss, accuracy = self.model_step(train_data, target, lengths)
        loss.backward()
        self.opt.step()
        if not self.scheduler is None:
            self.scheduler.step()

        return loss.item(), accuracy.item()

    def eval_step(self, validation_data: torch.tensor, target: torch.tensor, lengths: torch.tensor):
        '''
        Return is a tuple of (loss, accuracy)
        '''
        self.model.eval()
        loss, accuracy = self.model_step(validation_data, target, lengths)
        return loss, accuracy

    def model_step(self, input: torch.tensor, target: torch.tensor, lengths: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        '''
        Return is a tuple of (loss, accuracy)
        '''
        out = self.model(input, lengths)

        loss = self.loss(out.transpose(2, 1), target)

        accuracy = torch.mean(
            (self.model.get_predicted_seq_from_output(out) == target).float())

        return loss, accuracy

    def perform_eval_on_dataloader(self, dataloader: DataLoader) -> Tuple[float, float]:
        all_loss = []
        all_accuracy = []
        all_batch_size = []
        for _, (_, data, target, lengths) in enumerate(dataloader):
            data, target, lengths = data.to(
                self.device), target.to(self.device), lengths.to(self.device)
            loss, accuracy = self.eval_step(
                data, target, lengths)
            all_loss.append(loss)
            all_accuracy.append(accuracy)
            all_batch_size.append(len(lengths))

        mean_accuracy = (torch.tensor(
            all_accuracy) * torch.tensor(all_batch_size)).sum() / len(dataloader.dataset)
        mean_loss = (torch.tensor(all_loss) *
                     torch.tensor(all_batch_size)).sum() / len(dataloader.dataset)

        print(
            "=================================================" + "\n" +
            "| Evaluation result | Accuracy [%.4f] | Loss [%.4f]" % (
                mean_accuracy, mean_loss) + "\n"
            "================================================="
        )
        return mean_loss.item(), mean_accuracy.item()

    def save_training_result(self, train_file: str):
        '''
        Save to SavedModels folder:
        '''
        saving_folder_path = os.path.join(pathlib.Path(
            train_file).parent, "SavedModels/{}".format(str(datetime.now())))

        # Create folder for saving
        os.makedirs(saving_folder_path, exist_ok= True)

        # Save training records
        records_saving_path = os.path.join(saving_folder_path, TrainingRecord.records_save_file_name)
        self.record.save_records_to_file(records_saving_path)

        # Save training figure
        figure_saving_path = os.path.join(saving_folder_path, TrainingRecord.figure_save_file_name )
        self.record.save_figure(figure_saving_path)

        # Save model
        model_saving_path = os.path.join(saving_folder_path, self.model.model_save_file_name)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'epoch': self.epoch,
            'steps': self.steps,
        }, model_saving_path)

        print(
            "============================"+"\n" +
            "| Model saved successfully |" + "\n" +
            "============================"
        )

        print(
            "============================"+"\n" +
            "| Model saved folder: %s " % (saving_folder_path) + "\n" +
            "============================"
        )

    def load_trained_model(self, folder_path: str, load_optimizer: bool):
        figure_loading_path = os.path.join(folder_path, TrainingRecord.records_save_file_name)
        self.record.load_records(figure_loading_path)

        model_loading_path = os.path.join(folder_path, self.model.model_save_file_name)
        checkpoint = torch.load(model_loading_path)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if load_optimizer:
            self.opt.load_state_dict(checkpoint['optimizer_state_dict'])

        self.epoch = checkpoint['epoch']
        self.steps = checkpoint['steps']

        print(
            "============================="+"\n" +
            "| Model loaded successfully |" + "\n" +
            "============================="
        )

    def reset_epoch(self):
        self.epoch = 0

    def reset_steps(self):
        self.steps = 0
