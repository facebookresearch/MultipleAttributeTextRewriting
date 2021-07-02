# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from torch import nn
import torch


class MultiAttrDiscriminator(nn.Module):

    DIS_ATTR = ['input_dim', 'dis_layers', 'dis_hidden_dim', 'dis_dropout']

    def __init__(self, params):
        """
        Discriminator initialization.
        """
        super(MultiAttrDiscriminator, self).__init__()

        self.attr_values = params.attr_values
        self.input_dim = params.hidden_dim if params.attention and not params.transformer else params.hidden_dim
        self.dis_layers = params.dis_layers
        self.dis_hidden_dim = params.dis_hidden_dim
        self.dis_dropout = params.dis_dropout

        layers = []
        for i in range(self.dis_layers):
            if i == 0:
                input_dim = self.input_dim
                input_dim *= (2 if params.attention and not params.dis_input_proj else 1)
            else:
                input_dim = self.dis_hidden_dim
            layers.append(nn.Linear(input_dim, self.dis_hidden_dim))
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
        self.shared_layers = nn.Sequential(*layers)
        output_layers = []
        for attr_range in self.attr_values.values():
            output_layers.append(nn.Linear(self.dis_hidden_dim, len(attr_range)))
        self.output_layers = nn.ModuleList(output_layers)

    def forward(self, input):
        h = self.shared_layers(input)
        return [output_layer(h) for output_layer in self.output_layers]


class MultiAttrDiscriminatorLSTM(nn.Module):

    DIS_ATTR = ['input_dim', 'dis_layers', 'dis_hidden_dim', 'dis_dropout', 'lstm_dim', 'lstm_layers']

    def __init__(self, params):
        """
        Discriminator initialization.
        """
        super(MultiAttrDiscriminatorLSTM, self).__init__()

        self.attr_values = params.attr_values
        self.input_dim = params.hidden_dim if not params.attention else params.emb_dim
        assert params.disc_lstm_dim is not None
        assert params.disc_lstm_layers is not None
        self.lstm_dim = params.disc_lstm_dim
        self.lstm_layers = params.disc_lstm_layers
        self.dis_layers = params.dis_layers
        self.dis_hidden_dim = params.dis_hidden_dim
        self.dis_dropout = params.dis_dropout

        self.aggregator = nn.LSTM(
            self.input_dim, self.lstm_dim // 2, bidirectional=True,
            num_layers=self.lstm_layers
        )

        layers = []
        for i in range(self.dis_layers):
            if i == 0:
                input_dim = self.lstm_dim
                input_dim *= (2 if params.attention and not params.dis_input_proj else 1)
            else:
                input_dim = self.dis_hidden_dim
            layers.append(nn.Linear(input_dim, self.dis_hidden_dim))
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
        self.shared_layers = nn.Sequential(*layers)
        output_layers = []
        for attr_range in self.attr_values.values():
            output_layers.append(nn.Linear(self.dis_hidden_dim, len(attr_range)))
        self.output_layers = nn.ModuleList(output_layers)

    def forward(self, input, lengths):
        _, sorted_idx = torch.sort(lengths, descending=True)
        _, rev_sorted_idx = torch.sort(sorted_idx)
        sorted_input = input.index_select(1, sorted_idx)
        sorted_lengths = lengths.index_select(0, sorted_idx)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            sorted_input, sorted_lengths
        )
        _, (h_t, _) = self.aggregator(packed_input)
        h_t = torch.cat([h_t[-1], h_t[-2]], 1).index_select(0, rev_sorted_idx)
        h = self.shared_layers(h_t)
        return [output_layer(h) for output_layer in self.output_layers]
