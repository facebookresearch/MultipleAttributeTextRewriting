# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import torch
from torch import nn
from torch.nn import functional as F

from ..utils import get_mask, reload_model


logger = getLogger()


class ConvolutionalClassifier(nn.Module):

    CLF_ATTR = ['attributes', 'attr_values', 'n_words', 'pad_index', 'emb_dim', 'n_layers', 'n_kernels', 'kernel_size', 'dropout', 'dico']

    def __init__(self, params, dico):
        """
        Build a convolutional classifier.
        """
        super(ConvolutionalClassifier, self).__init__()

        # model parameters
        self.attributes = params.attributes
        self.attr_values = params.attr_values
        self.n_words = params.n_words
        self.pad_index = params.pad_index
        self.emb_dim = params.emb_dim
        self.n_layers = params.clf_n_layers
        self.n_kernels = params.clf_n_kernels
        self.kernel_size = params.clf_kernel_size
        self.dropout = params.clf_dropout
        self.dico = dico
        assert 0 <= self.dropout < 1
        assert self.kernel_size % 2 == 1
        assert self.n_words == len(self.dico)

        # embedding layer
        self.embeddings = nn.Embedding(self.n_words, self.emb_dim, padding_idx=self.pad_index)
        nn.init.normal_(self.embeddings.weight, 0, 0.1)
        nn.init.constant_(self.embeddings.weight[self.pad_index], 0)

        # convolutional layers
        self.conv_layers = nn.ModuleList([
            nn.Sequential(*[
                nn.Conv2d(
                    in_channels=1,
                    out_channels=self.n_kernels,
                    kernel_size=(self.kernel_size, self.n_kernels if i > 0 else self.emb_dim),
                    padding=(self.kernel_size // 2, 0)
                ),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ]) for i in range(self.n_layers)
        ])

        # projection layer
        self.n_labels = sum([len(x) for x in self.attr_values.values()])
        self.proj_layer = nn.Linear(self.n_kernels, self.n_labels)

    def forward(self, sent, lengths):
        """
        Take as input:
            - sent (of size (slen, bs))
            - lengths (of size (bs,))
        Return:
            - a vector of size  (bs, n_labels) with label probabilities for each attribute for each sentence
        """
        slen, bs = sent.size()
        assert lengths.max() == slen
        sent = sent.transpose(0, 1)

        # embeddings
        x = self.embeddings(sent).unsqueeze(1)                      # (bs, 1, slen, emb_dim)
        x = F.dropout(x, p=self.dropout, training=self.training)    # (bs, 1, slen, emb_dim)

        # convolutional layers
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)               # (bs, n_kernels, slen, 1)
            if i < len(self.conv_layers) - 1:
                x = x.transpose(1, 3)  # (bs, 1, slen, n_kernels)

        # mask out padding tokens
        mask = get_mask(lengths, True, self.n_kernels, batch_first=True).to(x.device)  # (bs, slen, n_kernels)
        mask = mask.transpose(1, 2).unsqueeze(3).contiguous()                          # (bs, n_kernels, slen, 1)
        x = x.masked_fill(1 - mask, -1e12)

        # projection layer
        x = x.squeeze(3)        # (bs, n_kernels, slen)
        x = x.max(2)[0]         # (bs, n_kernels)
        x = self.proj_layer(x)  # (bs, n_labels)

        assert x.size() == (bs, self.n_labels)
        return x


def check_classifier_params(params):
    """
    Check classifier parameters.
    """
    assert 0 <= params.clf_dropout < 1
    assert params.clf_kernel_size % 2 == 1


def build_classifier_model(params, dico, cuda=True):
    """
    Build a classifier.
    """
    logger.info("============ Building classifier model ...")
    classifier = ConvolutionalClassifier(params, dico)
    logger.info("")

    # cuda
    if cuda:
        classifier.cuda()

    # initialize the model with pretrained embeddings
    assert 'cpu_thread' not in params

    # reload classifier
    if params.reload_model != '':
        assert os.path.isfile(params.reload_model)
        logger.info("Reloading model from %s ..." % params.reload_model)
        reloaded = torch.load(params.reload_model)
        reload_model(classifier, reloaded['clf'], classifier.CLF_ATTR)

    # log models
    logger.info("============ Model summary")
    logger.info("Classifier: {}".format(classifier))
    logger.info("")

    return classifier
