from Models.ControllerModel import ControllerModel
import torch
import torch.nn as nn
from itertools import chain
from sklearn.preprocessing import StandardScaler

class BaseNNModel(nn.Module, ControllerModel):
    def __init__(self, feature_names, hidden_dim: list[int] = [], dropout: float = 0.1):
        super(BaseNNModel, self).__init__()
        self.feature_names = feature_names
        output_dim = 1

        all_dim = [len(self.feature_names)] + hidden_dim + [output_dim]

        all_layers = [[nn.Linear(all_dim[idx], all_dim[idx+1]), nn.BatchNorm1d(all_dim[idx+1]), nn.LeakyReLU(inplace=True), nn.Dropout(dropout)] if idx + 2 != len(all_dim) else [nn.Linear(all_dim[idx], all_dim[idx+1])] for idx in range(len(all_dim)-1)]
        # all_layers = [[nn.Linear(all_dim[idx], all_dim[idx+1])] if idx + 2 != len(all_dim) else [nn.Linear(all_dim[idx], all_dim[idx+1])] for idx in range(len(all_dim)-1)]

        self.model = nn.Sequential(
            *chain.from_iterable(all_layers),
            nn.Sigmoid()
        )

        self.mean_ = None
        self.var_ = None

        self.apply(BaseNNModel.weight_init)

    def forward(self, input: torch.tensor):
        return self.model(input)

    def data_forward(self, data):
        input,_ = data

        ######### Scale input #########
        input = self.normalize_input(input)

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
        return  torch.mean(((out > 0.5).squeeze().float() == target).float())

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
            nn.init.xavier_normal_(m.weight)
        elif type(m) in [nn.LSTM]:
            for name, value in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(value.data)
                if 'bias' in name:
                    value.data.normal_()

    def get_prediction_list_from_out(self, out, mask=None):
        return (out >0.5).float().tolist()

    def get_target_list_from_target(self, target, mask=None):
        return target.tolist()

    def generate_mask(self, target):
        return None

    def get_labels(self):
        return ["True","False"]

    def get_mean_and_variance(self, df, device):
        scaler = StandardScaler()
        scaler.fit(df[self.feature_names])
        self.mean_ = torch.tensor(scaler.mean_).float().to(device)
        self.var_ = torch.tensor(scaler.var_).float().to(device)

    def should_load_mean_and_vairance(self):
        return not self.has_mean_and_variance()

    def has_mean_and_variance(self,):
        return (not self.mean_ is None) and (not self.var_ is None) 

    def normalize_input(self, input):
        return (input - self.mean_) / torch.sqrt(self.var_)

    def reverse_normalize_input(self, input):
        return (input * torch.sqrt(self.var_)) + self.mean_


    
