from typing import Tuple
from torch.utils.data import DataLoader
from Controller.TrainingParameters import SelectableLrScheduler, TrainingParameters
from Utils.Constants import Constants
from enum import Enum
import torch
import torch.nn as nn
import torch.optim as optim
from Data.BPI2012Dataset import BPI2012Dataset
from Models.BaselineLSMTModel import BaselineLSTMModel
from Utils.Constants import Constants
from Controller.TrainingParameters import SelectableDatasets, SelectableModels, SelectableOptimizer, SelectableLoss
from CustomExceptions.Exceptions import NotSupportedError

from Utils.PrintUtils import print_peforming_task

class TrainingController:
    def __init__(self, dataset: SelectableDatasets, model: SelectableModels, optimizer: SelectableOptimizer, loss: SelectableLoss):
        # determine the device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Load dataset
        if (dataset == SelectableDatasets.BPI2012):
            self.dataset = BPI2012Dataset(TrainingParameters.bpi_2012_path)
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
        if (model == SelectableModels.BaseLineLSTMModel):
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
        if optimizer == SelectableOptimizer.Adam:
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
            raise NotSupportedError("Learning rate scheduler you selected is not supported");

        # Setting up loss

        if (loss == SelectableLoss.CrossEntropy):
            self.loss = nn.CrossEntropyLoss()
        else:
            raise NotSupportedError(
                "Loss function you selected is not supported")
        self.epoch = 0
        self.steps = 0

    def train(self, stop_epoch: int):
        self.model.to(self.device)
        while self.epoch < stop_epoch:
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
                    self.perform_eval_on_dataloader(
                        self.validation_data_loader)

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

        return loss, accuracy

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

    def perform_eval_on_dataloader(self, dataloader: DataLoader):
        all_loss = []
        all_accuracy = []
        all_batch_size = []
        for _, (_, data, target, lengths) in enumerate(dataloader):
            data, target, lengths = data.to(
                self.device), target.to(self.device), lengths.to(self.device)
            loss, accuracy = self.eval_step(
                data, target, lengths)
            all_loss.append(loss.item())
            all_accuracy.append(accuracy.item())
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

    def reset_epoch(self):
        self.epoch = 0

    def reset_steps(self):
        self.steps = 0
