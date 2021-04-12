
from operator import le
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BaselineLSTMModel(nn.Module):
    model_save_file_name = "BaselineLSTMModel.pt"

    def __init__(self, vocab_size: int, embedding_dim: int,  lstm_hidden: int, dropout: float, num_lstm_layers: int, paddingValue: int = 0):
        super(BaselineLSTMModel, self).__init__()

        self.emb = nn.Embedding(vocab_size, embedding_dim, paddingValue)
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden, batch_first=True,
                            dropout=dropout, num_layers=num_lstm_layers)

        self.output_net = nn.Sequential(
            nn.Linear(lstm_hidden, vocab_size),
        )

        # The trainable init h0 and c0 in LSTM.
        self.h0 = nn.Parameter(torch.randn(num_lstm_layers, 1, lstm_hidden))
        self.c0 = nn.Parameter(torch.randn(num_lstm_layers, 1, lstm_hidden))

        self.apply(BaselineLSTMModel.weight_init)

    def forward(self, input: torch.tensor, lengths: np.ndarray = None, prev_hidden_states: Tuple[torch.tensor] = None) -> Tuple[torch.tensor, Tuple[torch.tensor, torch.tensor]]:
        '''
        Input size: (B ,S)
        Output size: (B, S, vocab_size), ((num))
        Preveious state size: (num_lstm_layers, batch_szie, lstm_hidden)
        '''

        batch_size = input.size(0)

        # Prepare hidden state input
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

        # input (B, S)
        out = self.emb(input)  # ( B, S, F )

        if not lengths is None:
            out = pack_padded_sequence(out, lengths=lengths, batch_first=True)
            out, (h_out, c_out) = self.lstm(
                out, input_hidden_state)  # ( B, S, F)
            out, _ = pad_packed_sequence(out, batch_first=True)
        else:
            out, (h_out, c_out) = self.lstm(
                out, input_hidden_state)  # ( B, S, F)

        out = self.output_net(out)  # (B, S, vocab_size)
        return out, (h_out, c_out)

    def predict_next(self, input: torch.tensor, lengths: torch.tensor = None, previous_hidden_state: Tuple[torch.tensor, torch.tensor] = None, use_argmax: bool = False):
        self.eval()
        batch_size = input.size(0)  # (B, S)

        out, hidden_out = self.forward(
            input, prev_hidden_states=previous_hidden_state)  # (B, S, vocab_size)

        # Get the last output from each seq
        # len - 1 to get the index,
        # a len == 80 seq, will only have index 79 as the last output (from the 79 input)
        final_index = lengths - 1
        out = out[torch.arange(batch_size), final_index, :]  # (B, Vocab)
        out = F.softmax(out, dim=-1)  # (B, vocab_size)
        if (use_argmax):
            out = torch.argmax(out, dim=-1)  # (B)
        else:
            out = torch.multinomial(out, num_samples=1).squeeze()  # (B)

        return out, hidden_out

    def predict_next_n(self, input: torch.tensor, n: int, lengths: torch.tensor = None, use_argmax: bool = False):
        # Unpadded the input
        predicted_list = [[i.item() for i in l if i != 0] for l in input]

        # Initialise hidden state
        hidden_state = None
        for _ in range(n):
            # Predict
            predicted, hidden_state = self.predict_next(input=input, lengths=lengths,
                                                        previous_hidden_state=hidden_state, use_argmax=use_argmax)

            # Add predicted to unpadded.
            predicted_list = [u + [p.item()]
                              for u, p in zip(predicted_list, predicted)]

            # Assign for the next loop input, since tensor use reference, we won't use too much memory for it.
            input = predicted.unsqueeze(-1)
            lengths = torch.ones_like(lengths)

        return predicted_list

    def predict_next_till_eos(self, input: torch.tensor, lengths: torch.tensor, eos_idx: int, use_argmax: bool = False):
        # List for input data
        input_list = [[i.item() for i in l if i != 0] for l in input]

        print('Total cases: %d' % (len(input_list)))

        # List that prediction has been finished.
        predicted_list = [None] * len(input_list)

        # Initialise hidden state
        hidden_state = None
        while len(input_list) > 0:
            print("Before feeding")
            print(input)
            # Predict
            predicted, hidden_state = self.predict_next(input=input, lengths=lengths,
                                                        previous_hidden_state=hidden_state, use_argmax=use_argmax)
            
            # Check if it's 0-d tensor
            if (predicted.size() == ()):
                predicted = predicted.unsqueeze(0)

            for idx,  (il, p) in enumerate(zip(input_list, predicted)):
                # Append the predicted value
                p_v = p.item()
                input_list[idx] = il + [p_v]

                if (p_v == eos_idx):
                    print("Remain cases at start : %d" % (len(input_list)))
                    print("Remain predicted at start: %d" % (len(predicted)))
                    # Create index mapper (Mapping the input_list  to predicted_list)
                    idx_mapper = [idx for idx, pl in enumerate(
                        predicted_list) if pl is None]

                    # Assign to predicted_list (Remove from input list)
                    idx_in_predicted_list = idx_mapper[idx]
                    predicted_list[idx_in_predicted_list] = input_list.pop(idx)

                    batch_size = len(predicted)
                    # Remove instance from the lengths
                    lengths = lengths[torch.arange(batch_size) != idx]

                    # Remove instance from next input
                    predicted = predicted[torch.arange(batch_size) != idx, ]

                    # Remove the hidden state to enable next inter
                    h0 = hidden_state[0][:, torch.arange(batch_size) != idx, :]
                    c0 = hidden_state[1][:, torch.arange(batch_size) != idx, :]
                    hidden_state = (h0, c0)

                    print("Remain cases: %d" % (len(input_list)))
                    print("Remain predicted %d" % (len(predicted)))

                    if (len(predicted) == 0 and len(input_list) == 0):
                        break

            # Assign for the next loop input, since tensor use reference, we won't use too much memory for it.
            input = predicted.unsqueeze(-1)
            lengths = torch.ones_like(lengths)

        return predicted_list

    @staticmethod
    def weight_init(m) -> None:
        '''
        Initialising the weihgt
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
        Print how many parameters in the model
        '''
        return sum([param.nelement() for param in self.parameters()])
