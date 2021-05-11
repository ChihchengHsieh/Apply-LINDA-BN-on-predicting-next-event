
from Models.ControllerModel import ControllerModel
from Utils.VocabDict import VocabDict
import json
import os
import pathlib
import sys
from Parameters.TrainingParameters import TrainingParameters
from torch.utils.data.dataloader import DataLoader
from Data.PredictingJsonDataset import PredictingJsonDataset
from Utils.Constants import Constants
from Utils.PrintUtils import print_big, print_peforming_task, print_percentages, print_taks_done
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BaselineLSTMModel_V2(nn.Module, ControllerModel):
    '''
    LSTM model for predicing sequetial data.
    '''

    def __init__(self, vocab: VocabDict, embedding_dim: int,  lstm_hidden: int, dropout: float, num_lstm_layers: int):
        super(BaselineLSTMModel_V2, self).__init__()
        self.vocab = vocab
        self.emb = nn.Embedding(len(vocab), embedding_dim, self.vocab.padding_index())
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden, batch_first=True,
                            dropout=dropout, num_layers=num_lstm_layers)

        self.batchnorm = nn.BatchNorm1d(num_features=lstm_hidden)
        self.output_net = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, len(vocab)),
        )

        ############ The trainable init h0 and c0 in LSTM ############
        self.h0 = nn.Parameter(torch.zeros(num_lstm_layers, 1, lstm_hidden))
        self.c0 = nn.Parameter(torch.zeros(num_lstm_layers, 1, lstm_hidden))


        ############ Initialise weigth ############
        self.apply(self.weight_init)

    def forward(self, input: torch.tensor, lengths: np.ndarray = None, prev_hidden_states: Tuple[torch.tensor] = None) -> Tuple[torch.tensor, Tuple[torch.tensor, torch.tensor]]:
        '''
        Input size: (B ,S)
        [input]: input traces. (B,S)
        [lengths]: length of traces. (B)
        [previous_hidden_state]: hidden state in last time step, should be (h_, c_) ===Size===> ((num_layers, batch_size, lstm_hidden), (num_layers, batch_size, lstm_hidden))
        --------------
        return: output, hidden_state ===Size===> (B, S, vocab_size), ((num_layers, batch_size, lstm_hidden), (num_layers, batch_size, lstm_hidden))
        '''

        batch_size = input.size(0)

        ############ Prepare hidden state input ############
        if not prev_hidden_states is None:
            if (len(prev_hidden_states)) != 2:
                raise Exception("The length of given previous hidden state is not correct, expected %d, but get %d" % (
                    2, len(prev_hidden_states)))
            expected_previous_state_size = (
                self.lstm.num_layers, batch_size, self.lstm.hidden_size)
            if prev_hidden_states[0].size() != expected_previous_state_size:
                raise Exception("The expected size from previous state is %s, the input has size %s" % (
                    str(expected_previous_state_size), str(tuple(prev_hidden_states[0].size()))))

            if prev_hidden_states[1].size() != expected_previous_state_size:
                raise Exception("The expected size from previous state is %s, the input has size %s" % (
                    str(expected_previous_state_size), str(tuple(prev_hidden_states[1].size()))))
            input_hidden_state = prev_hidden_states
        else:
            input_hidden_state = (self.h0.repeat(
                1, batch_size, 1), self.c0.repeat(1, batch_size, 1))

        ############ Embedding layer ############
        out = self.emb(input)  # ( B, S, F )


        ############ LSTM ############
        if not lengths is None:
            out = pack_padded_sequence(
                out, lengths=lengths.cpu(), batch_first=True)
            out, (h_out, c_out) = self.lstm(
                out, input_hidden_state)  # ( B, S, F)
            out, _ = pad_packed_sequence(out, batch_first=True)
        else:
            out, (h_out, c_out) = self.lstm(
                out, input_hidden_state)  # ( B, S, F)

        ############ BatchNorm and last NN ############
        out = self.batchnorm(out.transpose(2, 1)).transpose(2, 1)  # (B, F, S)
        out = F.softmax(self.output_net(out), dim=-1)  # (B, S, vocab_size)

        return out, (h_out, c_out)

    def predict_next(self, input: torch.tensor, lengths: torch.tensor = None, previous_hidden_state: Tuple[torch.tensor, torch.tensor] = None, use_argmax: bool = False):
        '''
        Predict next activity.
        [input]: input traces.
        [lengths]: length of traces.
        [previous_hidden_state]: hidden state in last time step, should be (h_, c_)
        [use_argmax]: \n
        - True -> Select the vocab with largest possibility.\n
        - False -> the next prediction will be sampled from the distribution.\n
        -------------
        return: tuple(output, (h_out, c_out)).
        '''
        self.eval()
        batch_size = input.size(0)  # (B, S)

        out, hidden_out = self.forward(
            input, prev_hidden_states=previous_hidden_state)  # (B, S, vocab_size)

        ############ Get next activity ############
        # Get the last output from each seq
        # len - 1 to get the index,
        # a len == 80 seq, will only have index 79 as the last output (from the 79 input)
        final_index = lengths - 1
        out = out[torch.arange(batch_size), final_index, :]  # (B, Vocab)

        if (use_argmax):
            ############ Get the one with largest possibility ############
            out = torch.argmax(out, dim=-1)  # (B)
            # TODO: Testing value, need to delete
            self.argmax_out = out
        else:
            ############ Sample from distribution ############
            out = torch.multinomial(out, num_samples=1).squeeze(1)  # .squeeze()  # (B)

        return out, hidden_out

    def predict_next_n(self, input: torch.tensor, n: int, lengths: torch.tensor = None, use_argmax: bool = False)-> list[list[int]]:
        '''
        peform prediction n times.\n
        [input]: input traces
        [n]: number of steps to peform prediction.
        [lengths]: lengths of traces
        [use_argmax]: \n
        - True -> Select the vocab with largest possibility.\n
        - False -> the next prediction will be sampled from the distribution.\n
        -------------
        return: predicted list.
        '''
        ############ Unpadded input to get current taces ############
        predicted_list = [[i.item() for i in l if i != 0] for l in input]

        ############ Initialise hidden state ############
        hidden_state = None
        for i in range(n):
            ############ Predict############
            predicted, hidden_state = self.predict_next(input=input, lengths=lengths,
                                                        previous_hidden_state=hidden_state, use_argmax=use_argmax)

            ############ Add predicted to current traces ############
            predicted_list = [u + [p.item()]
                              for u, p in zip(predicted_list, predicted)]

            ############ Prepare for next step #########################################################################
            # Assign for the next loop input, since tensor use reference, we won't use too much memory for it.
            # And, we only use last step and the hidden state for predicting next.
            ############################################################################################################
            input = predicted.unsqueeze(-1)
            lengths = torch.ones_like(lengths)

        return predicted_list

    def predict_next_till_eos(self, input: torch.tensor, lengths: torch.tensor, eos_idx: int, use_argmax: bool = False, max_predicted_lengths=1000) -> list[list[int]]:
        '''
        pefrom predicting till <EOS> token show up.\n
        [input]: input traces
        [lengths]: lengths of traces
        [eos_idx]: index of <EOS> token
        [use_argmax]: \n
        - True -> Select the vocab with largest possibility.\n
        - False -> the next prediction will be sampled from the distribution.\n
        [max_predicted_lengths]: used to restricted the maximum step when n_step is None.
        -------------
        return: predicted list.
        '''

        ############ List for input data ############
        input_list = [[i.item() for i in l if i != 0] for l in input]

        ############ List that prediction has been finished ############
        predicted_list = [None] * len(input_list)

        ############ Initialise hidden state ############
        hidden_state = None
        while len(input_list) > 0:
            ############ Predict ############
            predicted, hidden_state = self.predict_next(input=input, lengths=lengths,
                                                        previous_hidden_state=hidden_state, use_argmax=use_argmax)

            ############ Check if it's 0-d tensor ############
            if (predicted.size() == ()):
                predicted = predicted.unsqueeze(0)

            for idx,  (il, p) in enumerate(zip(input_list, predicted)):
                ############ Append predicted value ############
                p_v = p.item()
                input_list[idx] = il + [p_v]

                if (p_v == eos_idx or len(input_list[idx]) > max_predicted_lengths):
                    ############ Create index mapper (Mapping the input_list to predicted_list) ############
                    idx_mapper = [idx for idx, pl in enumerate(
                        predicted_list) if pl is None]

                    ############ Assign to predicted_list (Remove from input list) ############
                    idx_in_predicted_list = idx_mapper[idx]
                    predicted_list[idx_in_predicted_list] = input_list.pop(idx)

                    batch_size = len(predicted)
                    ############ Remove instance from the lengths ############
                    lengths = lengths[torch.arange(batch_size) != idx]

                    ############ Remove instance from next input ############
                    predicted = predicted[torch.arange(batch_size) != idx, ]

                    ############ Remove the hidden state to enable next inter ############
                    h0 = hidden_state[0][:, torch.arange(batch_size) != idx, :]
                    c0 = hidden_state[1][:, torch.arange(batch_size) != idx, :]
                    hidden_state = (h0, c0)

                    if (len(predicted) == 0 and len(input_list) == 0):
                        break

            ############################################################
            # Assign for the next loop input, since tensor use reference, we won't use too much memory for it.
            ############################################################
            input = predicted.unsqueeze(-1)
            lengths = torch.ones_like(lengths)

        return predicted_list

    def weight_init(self, m) -> None:
        '''
        Initialising the weihgt.

        use in the model like this: 
        \t self.apply(self.weight_init)
        '''
        if type(m) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.Conv1d]:
            nn.init.kaiming_normal_(m.weight, 0.2, nonlinearity='leaky_relu')
        elif type(m) in [nn.LSTM]:
            for name, value in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(value.data)
                if 'bias' in name:
                    value.data.normal_()

    def num_all_params(self,) -> int:
        '''
        return: how many parameters in the model
        '''
        return sum([param.nelement() for param in self.parameters()])

    def data_forward(
        self, data: tuple[np.ndarray, torch.Tensor, torch.Tensor, np.ndarray]
    ) -> torch.tensor:
        '''
        [data]: tuple of (case_id_array, padded_input, lengths_of_input, target)
        -----------
        return: output of the model.
        '''

        ############# Unpacked data #############
        _, input, lengths, _ = data

        out, _ = self.forward(input, lengths)

        return out

    def get_accuracy(self, out, target):
        '''
        Use argmax to get the final output, and get accuracy from it.
        [out]: output of model.
        [target]: target of input data.
        --------------
        return: accuracy value
        '''
        return torch.mean(
            torch.masked_select(
                (torch.argmax(out, dim=-1) == target), target > 0
            ).float()
        )

    def get_loss(self, loss_fn: callable, out, target):
        '''
        [loss_fn]: loss function to compute the loss.\n
        [out]: output of the model\n
        [target]: target of input data.\n

        ---------------------
        return: loss value
        '''
        return loss_fn(out.transpose(2, 1), target)

    
    def predict(
        self,
        input: torch.tensor,
        lengths: torch.tensor = None,
        n_steps: int = None,
        use_argmax=False,
        max_predicted_lengths = 50,
    )-> list[list[int]]:  

        '''
        [input]: tensor to predict\n
        [lengths]: lengths of input\n
        [n_step]: how many steps will be predicted. If n_step == None, model will
        repeat to predict till <EOS> token show.\n
        [use_argmax]: \n
        - True -> Select the vocab with largest possibility.\n
        - False -> the next prediction will be sampled from the distribution.\n
        [max_predicted_lengths]: used to restricted the maximum step when n_step is None.

        ----------------------------
        return: predicted list.

        '''
        if not n_steps is None:
            ######### Predict for next n activities #########
            predicted_list = self.predict_next_n(
                input=input, lengths=lengths, n=n_steps, use_argmax=use_argmax
            )

        else:
            ######### Predict till <EOS> token #########
            '''
            This method has the risk of causing infinite loop,
            `max_predicted_lengths` is used for restricting this behaviour.
            '''
            predicted_list = self.predict_next_till_eos(
                input=input,
                lengths=lengths,
                eos_idx=self.vocab.vocab_to_index(Constants.EOS_VOCAB),
                use_argmax=use_argmax,
                max_predicted_lengths=max_predicted_lengths,
            )

        return predicted_list

    def predicting_from_list_of_idx_trace(
        self, data: list[list[int]], n_steps: int = None, use_argmax=False
    ):  
        '''
        [data]: 2D list of token indexs.
        [n_step]: how many steps will be predicted. If n_step == None, model will
        repeat to predict till <EOS> token show.\n
        [use_argmax]: \n
        - True -> Select the vocab with largest possibility.\n
        - False -> the next prediction will be sampled from the distribution.\n

        ----------------
        return predited 2d list of token indexs.
        '''

        ######### To sort the input by lengths and get lengths #########
        data, lengths = self.tranform_to_input_data_from_seq_idx(data)

        ######### Predict #########
        predicted_list = self.predict(
            data=data, lengths=lengths, n_steps=n_steps, use_argmax=use_argmax
        )

        return predicted_list


    def predicting_from_list_of_vacab_trace(
        self, data: list[list[str]], n_steps: int = None, use_argmax=False
    ):
        '''
        [data]: 2D list of tokens.
        [n_step]: how many steps will be predicted. If n_step == None, model will
        repeat to predict till <EOS> token show.\n
        [use_argmax]: \n
        - True -> Select the vocab with largest possibility.\n
        - False -> the next prediction will be sampled from the distribution.\n

        ----------------
        return predited 2d list of tokens.
        '''

        ######### Transform to index #########
        data = [self.list_of_vocab_to_index(l) for l in data]

        ######### Predict #########
        predicted_list = self.predicting_from_list_of_idx_trace(
            data=data, n_steps=n_steps, use_argmax=use_argmax
        )
        
        ######### Tranform back to vocab #########
        predicted_list = [
            self.list_of_index_to_vocab(l) for l in predicted_list
        ]

        return predicted_list

    def load_json_for_predicting(
        self,
        path: str,
        device: torch.device,
        n_steps: int = None,
        use_argmax=False,
        max_predicted_lengths: int = 50
    )-> str:
        '''
        Load a json file, and predict the traces for it.\n

        Expected file structure:
        {
            "__caseid__": [ vocab_#1, vocab_#2 ]
        }
        
        [path]: path of the json file
        [device]: torch.device that the prediction will run on. (Make sure the model is loaded to this device as well)
        [n_step]: how many steps will be predicted. If n_step == None, model will
        repeat to predict till <EOS> token show.\n
        [use_argmax]: \n
        - True -> Select the vocab with largest possibility.\n
        - False -> the next prediction will be sampled from the distribution.\n
        [max_predicted_lengths]: used to restricted the maximum step when n_step is None.

        -------------
        return: result file path.
        '''
        print_peforming_task("Dataset Loading")

        ##########################################
        # Transform the json to dataloader
        ##########################################
        
        p_dataset = PredictingJsonDataset(
            self.vocab, predicting_file_path=path, device=device)

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


        ##########################################
        #   Start prediction
        ##########################################

        for _, (caseids, data, lengths) in enumerate(p_loader):

            data, lengths = data.to(self.device), lengths.to(self.device)
            predicted_list = self.predict(
                data=data, lengths=lengths, n_steps=n_steps, use_argmax=use_argmax,
                max_predicted_lengths= max_predicted_lengths
            )

            for k, v in zip(caseids, predicted_list):
                predicted_dict[k] = self.vocab.list_of_index_to_vocab(v)

            sys.stdout.write("\r")
            print_percentages(
                prefix="Predicting cases",
                percentage=len(predicted_dict) / len(p_dataset),
            )
            sys.stdout.flush()

        print_taks_done("Predicting")

        ##################################
        #   Save predicted result
        ##################################
        saving_path = pathlib.Path(path)
        saving_file_name = pathlib.Path(path).stem + "_result.json"
        saving_dest = os.path.join(saving_path.parent, saving_file_name)
        with open(saving_dest, "w") as output_file:
            json.dump(predicted_dict, output_file, indent="\t")

        print_big("Predition result has been save to: %s" % (saving_dest))

        return saving_dest

    
    def get_prediction_list_from_out(self, out, mask=None):
        predicted = torch.argmax(out, dim=-1)  # (B, S)
        selected_predictions = torch.masked_select(
            predicted, mask)

        return selected_predictions.tolist()

    def get_target_list_from_target(self, target, mask=None):
        selected_targets = torch.masked_select(
            target, mask
        )
        return selected_targets.tolist()

    def generate_mask(self, target):
        return target > 0

    def get_labels(self):
        return self.vocab.vocab_dict.keys()

    def get_mean_and_variance(self,df, device):
        pass
    
    def should_load_mean_and_vairance(self):
        return False

    def has_mean_and_variance(self,):
        return False 


    

    
