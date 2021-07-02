# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
from functools import reduce
from operator import mul
import torch
from torch import nn
from torch.nn import functional as F

from ..utils import reload_model


logger = getLogger()


class LM(nn.Module):

    LM_ATTR = ['attributes', 'attr_values', 'n_words', 'emb_dim', 'hidden_dim', 'bias', 'dropout', 'bos_attr', 'bias_attr', 'dico', 'bos_index', 'eos_index', 'pad_index']

    def __init__(self, params, dico):
        """
        Language model initialization.
        """
        super(LM, self).__init__()

        # model parameters
        self.attributes = params.attributes
        self.attr_values = params.attr_values
        self.n_words = params.n_words
        self.emb_dim = params.emb_dim
        self.hidden_dim = params.lm_hidden_dim
        self.bias = params.lm_bias
        self.dropout = params.lm_dropout
        self.bos_attr = params.lm_bos_attr
        self.bias_attr = params.lm_bias_attr
        self.dico = dico
        assert 0 <= self.dropout < 1
        assert self.n_words == len(self.dico)
        assert self.bos_attr in ['', 'avg', 'cross']
        assert self.bias_attr in ['', 'avg', 'cross']

        # indexes
        self.bos_index = params.bos_index
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index

        # attribute embeddings / bias
        if self.bos_attr != '' or self.bias_attr != '':
            self.register_buffer('attr_offset', params.attr_offset.clone())
            self.register_buffer('attr_shifts', params.attr_shifts.clone())
        if self.bos_attr != '':
            n_bos_attr = sum(params.n_labels) if self.bos_attr == 'avg' else reduce(mul, params.n_labels, 1)
            self.bos_attr_embeddings = nn.Embedding(n_bos_attr, self.emb_dim)
        if self.bias_attr != '':
            n_bias_attr = sum(params.n_labels) if self.bias_attr == 'avg' else reduce(mul, params.n_labels, 1)
            self.bias_attr_embeddings = nn.Embedding(n_bias_attr, self.n_words)

        # embedding layers
        self.embeddings = nn.Embedding(self.n_words, self.emb_dim, padding_idx=self.pad_index)
        nn.init.normal_(self.embeddings.weight, 0, 0.1)
        nn.init.constant_(self.embeddings.weight[self.pad_index], 0)

        # LSTM layers
        self.lstm = nn.LSTM(self.emb_dim, self.hidden_dim, num_layers=1, dropout=self.dropout)

        # projection layers
        self.proj = nn.Linear(self.hidden_dim, self.n_words, bias=self.bias)

    def get_bos_attr(self, attr):
        """
        Generate beginning of sentence attribute embedding.
        """
        if self.bos_attr == 'avg':
            return self.bos_attr_embeddings(attr).mean(1)
        if self.bos_attr == 'cross':
            return self.bos_attr_embeddings(((attr - self.attr_offset[None]) * self.attr_shifts[None]).sum(1))
        assert False

    def get_bias_attr(self, attr):
        """
        Generate attribute bias.
        """
        if self.bias_attr == 'avg':
            return self.bias_attr_embeddings(attr).mean(1)
        if self.bias_attr == 'cross':
            return self.bias_attr_embeddings(((attr - self.attr_offset[None]) * self.attr_shifts[None]).sum(1))
        assert False

    def forward(self, sent, lengths, attr):
        """
        Input:
            - LongTensor of size (slen, bs), word indices
            - LongTensor of size (bs,), sentence lengths
        Output:
            - FloatTensor of size (slen, bs, n_words),
              representing the score for each output word of being the next word
        """
        slen, bs = sent.size()
        assert lengths.max() == slen and lengths.size(0) == bs
        assert attr.size() == (bs, len(self.attributes))

        # embeddings
        embeddings = self.embeddings(sent)
        if self.bos_attr != '':
            embeddings[0] = self.get_bos_attr(attr)
        embeddings = F.dropout(embeddings, p=self.dropout, training=self.training)
        assert embeddings.size() == (slen, bs, self.emb_dim)

        # LSTM
        lstm_output, (_, _) = self.lstm(embeddings)
        assert lstm_output.size() == (slen, bs, self.hidden_dim)

        # word scores
        word_scores = self.proj(lstm_output)
        if self.bias_attr != '':
            word_scores = word_scores + self.get_bias_attr(attr)[None]
        return word_scores

    def generate(self, attr, max_len=200, temperature=-1):
        """
        Generate sentences from attributes.
        """
        assert temperature > 0 or temperature == -1

        bs = attr.size(0)
        cur_len = 1
        decoded = torch.LongTensor(max_len, bs).fill_(self.pad_index).to(attr.device)
        decoded[0] = self.bos_index
        h_c = None

        # decoding
        while cur_len < max_len:

            # previous word embeddings
            if cur_len == 1 and self.bos_attr != '':
                embeddings = self.get_bos_attr(attr)
            else:
                embeddings = self.embeddings(decoded[cur_len - 1])
            embeddings = F.dropout(embeddings, p=self.dropout, training=self.training)

            lstm_output, h_c = self.lstm(embeddings.unsqueeze(0), h_c)
            output = F.dropout(lstm_output, p=self.dropout, training=self.training).view(bs, self.hidden_dim)
            scores = self.proj(output)
            if self.bias_attr != '':
                scores = scores + self.get_bias_attr(attr)
            assert scores.size() == (bs, self.n_words)

            # select next words: sample or argmax
            if temperature > 0:
                next_words = torch.multinomial(F.softmax(scores / temperature, dim=1), 1).squeeze(1)
            else:
                next_words = scores.max(1)[1]
            assert next_words.size() == (bs,)
            decoded[cur_len] = next_words
            cur_len += 1

            # stop when there is a </s> in each sentence
            if decoded.eq(self.eos_index).sum(0).ne(0).sum() == bs:
                break

        # compute the length of each generated sentence, and
        # put some padding after the end of each sentence
        lengths = torch.LongTensor(bs).fill_(cur_len)
        for i in range(bs):
            for j in range(cur_len):
                if decoded[j, i] == self.eos_index:
                    if j + 1 < max_len:
                        decoded[j + 1:, i] = self.pad_index
                    lengths[i] = j + 1
                    break
            if lengths[i] == max_len:
                decoded[-1, i] = self.eos_index

        return decoded[:cur_len], lengths


def check_lm_params(params):
    """
    Check language model parameters.
    """
    assert 0 <= params.lm_dropout < 1


def build_language_model(params, dico, cuda=True):
    """
    Build a language model.
    """
    logger.info("============ Building language model ...")
    lm = LM(params, dico)
    logger.info("")

    # cuda
    if cuda:
        lm.cuda()

    # initialize the model with pretrained embeddings
    assert 'cpu_thread' not in params

    # reload language model
    if params.reload_model != '':
        assert os.path.isfile(params.reload_model)
        logger.info("Reloading model from %s ..." % params.reload_model)
        reloaded = torch.load(params.reload_model)
        reload_model(lm, reloaded['lm'], lm.LM_ATTR)

    # log models
    logger.info("============ Model summary")
    logger.info("Language model: {}".format(lm))
    logger.info("")

    return lm
