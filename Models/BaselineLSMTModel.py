
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BaselineLSTMModel(nn.Module):
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

    def forward(self, input: torch.tensor, lengths: np.ndarray = None) -> torch.tensor:
        '''
        Input size: (B ,S)
        Output size: (B, S, vocab_size)
        '''
        # input (B, S)
        batch_size = input.size(0)
        out = self.emb(input)  # ( B, S, F )

        if not lengths is None:
            out = pack_padded_sequence(out, lengths=lengths, batch_first=True)
            out, _ = self.lstm(out, (self.h0.repeat(
                1, batch_size, 1), self.c0.repeat(1, batch_size, 1)))  # ( B, S, F)
            out, _ = pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.lstm(out, (self.h0.repeat(
                1, batch_size, 1), self.c0.repeat(1, batch_size, 1)))  # ( B, S, F)

        out = self.output_net(out)  # (B, S, vocab_size)
        return out

    def argmax_prediction(self, input: torch.tensor, lengths: np.ndarray = None, onlyReturnFinalStep: bool = True) -> torch.tensor:
        seq_size = input.size(1)
        out = self.forward(input)  # (B, S, vocab_size)
        out = F.softmax(out, dim=-1)  # (B, S, vocab_size)
        out = torch.argmax(out, dim=-1)  # (B, S)
        if onlyReturnFinalStep:
            final_index = torch.tensor([l - 1 for l in lengths])
            final_out_mask = torch.gt(F.one_hot(final_index, seq_size), 0)
            out =  out.masked_select(final_out_mask) # (B)
        return out