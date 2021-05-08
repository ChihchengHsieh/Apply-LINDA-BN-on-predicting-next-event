from Models.ControllerModel import ControllerModel
import torch
import torch.nn as nn
from itertools import chain

class BaseNNModel(nn.Module, ControllerModel):
    def __init__(self, num_input_features, hidden_dim: list[int] = []):
        super(BaseNNModel, self).__init__()
        output_dim = 1

        all_dim = [num_input_features] + hidden_dim + [output_dim]

        all_layers = [[nn.Linear(all_dim[idx], all_dim[idx+1]), nn.BatchNorm1d(all_dim[idx+1]), nn.LeakyReLU(inplace=True), nn.Dropout(.2)] if idx + 2 != len(all_dim) else [nn.Linear(all_dim[idx], all_dim[idx+1])] for idx in range(len(all_dim)-1)]
        # all_layers = [[nn.Linear(all_dim[idx], all_dim[idx+1])] if idx + 2 != len(all_dim) else [nn.Linear(all_dim[idx], all_dim[idx+1])] for idx in range(len(all_dim)-1)]

        self.model = nn.Sequential(
            *chain.from_iterable(all_layers),
            nn.Sigmoid()
        )

        self.apply(BaseNNModel.weight_init)

    def forward(self, input: torch.tensor):
        return self.model(input)

    def data_forward(self, data):
        input,_ = data
        out = self.forward(input)
        return out

    def get_accuracy(self, out, target):
        '''
        Use argmax to get the final output, and get accuracy from it.
        [out]: output of model.
        [target]: target of input data.
        --------------
        return: accuracy value
        '''
        return  torch.mean(((out > 0.5).float() == target).float())

    def get_loss(self, loss_fn: callable, out, target):
        '''
        [loss_fn]: loss function to compute the loss.\n
        [out]: output of the model\n
        [target]: target of input data.\n

        ---------------------
        return: loss value
        '''
        return loss_fn(out.squeeze(), target.squeeze())


    def num_all_params(self,) -> int:
        '''
        return how many parameters in the model
        '''
        return sum([param.nelement() for param in self.parameters()])

    @staticmethod
    def weight_init(m) -> None:
        '''
        Initialising the weihgt
        '''
        if type(m) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.Conv1d]:
            # nn.init.kaiming_normal_(m.weight, 0.2, nonlinearity='leaky_relu')
            nn.init.xavier_normal(m.weight)
        elif type(m) in [nn.LSTM]:
            for name, value in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(value.data)
                if 'bias' in name:
                    value.data.normal_()