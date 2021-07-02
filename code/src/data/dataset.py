# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import math
import numpy as np
import torch


logger = getLogger()


class Dataset(object):

    def __init__(self, sent, attr, pos, dico, params):

        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.unk_index = params.unk_index
        self.bos_index = params.bos_index
        self.batch_size = params.batch_size

        self.attributes = params.attributes
        self.attr_values = params.attr_values
        self.sent = sent
        self.attr = attr
        self.pos = pos
        self.dico = dico
        self.lengths = self.pos[:, 1] - self.pos[:, 0]
        self.is_parallel = False

        # check number of sentences / attributes
        assert len(self.pos) == (self.sent.numpy() == -1).sum()
        assert self.attr.size() == (len(self.pos), len(self.attributes))

        # remove empty sentences
        self.remove_empty_sentences()

        # sanity checks
        assert len(pos) == (sent[torch.from_numpy(pos[:, 1])] == -1).sum()  # check sentences indices
        assert -1 <= sent.min() < sent.max() < len(dico)                    # check dictionary indices
        assert self.lengths.min() > 0                                       # check empty sentences

        # attribute offset / shifts
        params.n_labels = [len(self.attr_values[a]) for a in self.attributes]
        params.attr_offset = torch.LongTensor(np.cumsum([0] + params.n_labels[:-1]))
        params.attr_shifts = torch.LongTensor(np.cumprod((params.n_labels[1:] + [1])[::-1])[::-1].copy())
        self.attr_offset = params.attr_offset
        self.attr_shifts = params.attr_shifts

    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        return len(self.pos)

    def batch_sentences(self, sentences, attributes):
        """
        Take as input a list of n sentences (torch.LongTensor vectors) and return
        a tensor of size (s_len, n) where s_len is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        lengths = torch.LongTensor([len(s) + 2 for s in sentences])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(self.pad_index)

        sent[0] = self.bos_index
        for i, s in enumerate(sentences):
            sent[1:lengths[i] - 1, i].copy_(s)
            sent[lengths[i] - 1, i] = self.eos_index

        return sent, lengths, attributes + self.attr_offset[None]

    def remove_empty_sentences(self):
        """
        Remove empty sentences.
        """
        init_size = len(self.pos)
        indices = np.arange(len(self.pos))
        indices = indices[self.lengths[indices] > 0]
        self.attr = self.attr[indices]
        self.pos = self.pos[indices]
        self.lengths = self.pos[:, 1] - self.pos[:, 0]
        logger.info("Removed %i empty sentences." % (init_size - len(indices)))

    def remove_long_sentences(self, max_len):
        """
        Remove sentences exceeding a certain length.
        """
        assert max_len > 0
        init_size = len(self.pos)
        indices = np.arange(len(self.pos))
        indices = indices[self.lengths[indices] <= max_len]
        self.attr = self.attr[indices]
        self.pos = self.pos[indices]
        self.lengths = self.pos[:, 1] - self.pos[:, 0]
        logger.info("Removed %i too long sentences." % (init_size - len(indices)))

    def select_data(self, a, b):
        """
        Only retain a subset of the dataset.
        """
        assert 0 <= a <= b <= len(self.pos)
        if a < b:
            self.attr = self.attr[a:b]
            self.pos = self.pos[a:b]
            self.lengths = self.pos[:, 1] - self.pos[:, 0]
        else:
            self.attr = torch.LongTensor()
            self.pos = torch.LongTensor()
            self.lengths = torch.LongTensor()

    def create_attr_idx(self):
        """
        Create attribute indexes.
        """
        self.attr_idx = []

        for i, attr in enumerate(self.attributes):
            attr_idx = []
            for j in range(len(self.attr_values[attr])):
                label_idx = np.arange(len(self.pos))
                label_idx = label_idx[self.attr[label_idx, i].numpy() == j]
                attr_idx.append(label_idx)
            self.attr_idx.append(attr_idx)
            assert len(self.pos) == sum([len(x) for x in attr_idx])

    def get_batches_iterator(self, batches):
        """
        Return a sentences iterator, given the associated sentence batches.
        """
        def iterator():
            for sentence_ids in batches:
                pos = self.pos[sentence_ids]
                sent = [self.sent[a:b] for a, b in pos]
                attr = self.attr[sentence_ids]
                yield self.batch_sentences(sent, attr)
        return iterator

    def get_iterator(self, shuffle, group_by_size=False, n_sentences=-1, attr_label=None):
        """
        Return a sentences iterator.
        """
        n_max = len(self.pos) if attr_label is None else len(self.attr_idx[attr_label[0]][attr_label[1]])
        n_sentences = n_max if n_sentences == -1 else n_sentences
        if n_sentences > n_max:
            logger.warning("Required %i sentences, but only %i were available!" % (n_sentences, n_max))
            n_sentences = n_max
        assert 0 < n_sentences <= n_max
        assert type(shuffle) is bool and type(group_by_size) is bool

        if shuffle:
            indices = np.random.permutation(n_max)[:n_sentences]
        else:
            indices = np.arange(n_sentences)

        if attr_label is not None:
            indices = self.attr_idx[attr_label[0]][attr_label[1]][indices]

        # group sentences by lengths
        if group_by_size:
            indices = indices[np.argsort(self.lengths[indices], kind='mergesort')]

        # create batches / optionally shuffle them
        batches = np.array_split(indices, math.ceil(len(indices) * 1. / self.batch_size))
        if shuffle:
            np.random.shuffle(batches)

        # return the iterator
        return self.get_batches_iterator(batches)
