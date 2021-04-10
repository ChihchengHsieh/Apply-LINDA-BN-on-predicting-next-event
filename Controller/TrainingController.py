from torch.utils.data import DataLoader
from Controller.TrainingParameters import TrainingParameters
from Utils.Constants import Constants
from enum import Enum
import torch
import torch.nn as nn
import torch.optim as optim
from Data.BPI2012Dataset import BPI2012Dataset
from Models.BaselineLSMTModel import BaselineLSTMModel
from Utils.Constants import Constants
from Controller.TrainingParameters import SelectableDatasets, SelectableModels, SelectableOptimizer, SelectableLoss

class NotSupportedError(Exception):
    pass

class TrainingController:
    def __init__(self, dataset: SelectableDatasets, model: SelectableModels, optimizer: SelectableOptimizer, loss: SelectableLoss):
    
        ## Load dataset
        if (dataset == SelectableDatasets.BPI2012):
            self.dataset = BPI2012Dataset(TrainingParameters.bpi_2012_path)
        else:
            raise NotSupportedError("Dataset you selected is not supported")

        self.data_loader = DataLoader(self.dataset, batch_size=32, shuffle=True, collate_fn= self.dataset.collate_fn)

        # Setting up model 
        if (model == SelectableModels.BaseLineLSTMModel):
            self.model = BaselineLSTMModel(
                vocab_size= self.dataset.vocab_size(),
                embedding_dim= TrainingParameters.BaselineLSTMModelParameters.embedding_dim,
                lstm_hidden= TrainingParameters.BaselineLSTMModelParameters.lstm_hidden,
                dropout= TrainingParameters.BaselineLSTMModelParameters.dropout,
                num_lstm_layers= TrainingParameters.BaselineLSTMModelParameters.num_lstm_layers,
                paddingValue= self.dataset.vocab_to_index(Constants.PAD_VOCAB),
            )
        else:
            raise NotSupportedError("Model you selected is not supported")

        # Setting up optimizer
        if optimizer == SelectableOptimizer.Adam:
            self.opt = optim.Adam(
                self.model.parameters(),
                lr =TrainingParameters.OptimizerParameters.learning_rate,
                                weight_decay= TrainingParameters.OptimizerParameters.l2
                                )
        else: 
             raise NotSupportedError("Optimizer you selected is not supported")

        # Setting up loss

        if (loss == SelectableLoss.CrossEntropy):
            self.loss = nn.CrossEntropyLoss()
        else:
            raise NotSupportedError("Loss function you selected is not supported")
        self.epoch = 0
        self.steps = 0

    def train(self, stop_epoch: int):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        while self.epoch < stop_epoch: 
            self.model.train()
            for _, (_, train, target, lengths) in enumerate(self.data_loader):
                train, target, lengths = train.to(device), target.to(device), lengths.to(device)
                loss, accuracy = self.train_step(train, target, lengths)
                self.steps += 1

                if self.steps % 5 == 0:
                    print('| Epoch [%d] | Step [%d] | lr [%.6f] | Loss: [%.4f] | Acc: [%.4f]|' % (self.epoch, self.steps, self.opt.param_groups[0]['lr'], loss, accuracy))

            self.epoch += 1

    def train_step(self, train, target, lengths):
        self.opt.zero_grad()
        loss, accuracy = self.model_step(train, target, lengths)
        loss.backward()
        self.opt.step()

        return loss, accuracy


    def model_step(self, train, target, lengths):

        out = self.model(train, lengths)

        loss = self.loss(out.transpose(2,1), target)

        accuracy = torch.mean((self.model.get_predicted_seq_from_output(out) == target).float())

        return loss, accuracy

    def reset_epoch(self):
        self.epoch = 0

    def reset_steps(self):
        self.steps = 0

    