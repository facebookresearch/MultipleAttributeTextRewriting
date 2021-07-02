# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import torch
from logging import getLogger


logger = getLogger()


BOS_WORD = '<s>'
EOS_WORD = '</s>'
PAD_WORD = '<pad>'
UNK_WORD = '<unk>'

SPECIAL_WORD = '<special%i>'
SPECIAL_WORDS = 10


class Dictionary(object):

    def __init__(self, id2word, word2id):
        assert len(id2word) == len(word2id)
        self.id2word = id2word
        self.word2id = word2id
        self.bos_index = word2id[BOS_WORD]
        self.eos_index = word2id[EOS_WORD]
        self.pad_index = word2id[PAD_WORD]
        self.unk_index = word2id[UNK_WORD]
        self.check_valid()

    def __len__(self):
        """Returns the number of words in the dictionary"""
        return len(self.id2word)

    def __getitem__(self, i):
        """
        Returns the word of the specified index.
        """
        return self.id2word[i]

    def __contains__(self, w):
        """
        Returns whether a word is in the dictionary.
        """
        return w in self.word2id

    def __eq__(self, y):
        """
        Compare this dictionary with another one.
        """
        self.check_valid()
        y.check_valid()
        if len(self.id2word) != len(y):
            return False
        return all(self.id2word[i] == y[i] for i in range(len(y)))

    def check_valid(self):
        """
        Check that the dictionary is valid.
        """
        assert self.bos_index == 0
        assert self.eos_index == 1
        assert self.pad_index == 2
        assert self.unk_index == 3
        assert all(self.id2word[4 + i] == SPECIAL_WORD % i for i in range(SPECIAL_WORDS))
        assert len(self.id2word) == len(self.word2id)
        for i in range(len(self.id2word)):
            assert self.word2id[self.id2word[i]] == i

    def index(self, word, no_unk=False):
        """
        Returns the index of the specified word.
        """
        if no_unk:
            return self.word2id[word]
        else:
            return self.word2id.get(word, self.unk_index)

    def prune(self, max_vocab):
        """
        Limit the vocabulary size.
        """
        assert max_vocab >= 1
        self.id2word = {k: v for k, v in self.id2word.items() if k < max_vocab}
        self.word2id = {v: k for k, v in self.id2word.items()}
        self.check_valid()

    @staticmethod
    def read_vocab(vocab_path):
        """
        Create a dictionary from a vocabulary file.
        """
        skipped = 0
        assert os.path.isfile(vocab_path), vocab_path
        word2id = {BOS_WORD: 0, EOS_WORD: 1, PAD_WORD: 2, UNK_WORD: 3}
        for i in range(SPECIAL_WORDS):
            word2id[SPECIAL_WORD % i] = 4 + i
        f = open(vocab_path, 'r', encoding='utf-8')
        for i, line in enumerate(f):
            if '\u2028' in line:
                skipped += 1
                continue
            line = line.rstrip().split()
            assert len(line) == 2, (i, line)
            assert line[0] not in word2id and line[1].isdigit(), (i, line)
            word2id[line[0]] = 4 + SPECIAL_WORDS + i - skipped  # shift because of extra words
        f.close()
        id2word = {v: k for k, v in word2id.items()}
        dico = Dictionary(id2word, word2id)
        logger.info("Read %i words from the vocabulary file." % len(dico))
        if skipped > 0:
            logger.warning("Skipped %i empty lines!" % skipped)
        return dico

    @staticmethod
    def index_data(txt_path, bin_path, dico, attr_list, attr_cols, attr_values):
        """
        Index sentences with a dictionary.
        """
        if os.path.isfile(bin_path):
            print("Loading data from %s ..." % bin_path)
            data = torch.load(bin_path)
            assert dico == data['dico']
            assert attr_values == data['attr_values']
            return data

        positions = []
        sentences = []
        attributes = []
        unk_words = {}
        count_empty_sentences = 0
        count_unknown_labels = 0
        label2id = {attr: {label: i for i, label in enumerate(labels)} for attr, labels in attr_values.items()}

        # index sentences
        f = open(txt_path, 'r', encoding='utf-8')
        for i, line in enumerate(f):
            if i % 100000 == 0 and i > 0:
                print(i)
            s = line.rstrip()
            # skip empty sentences
            if len(s) == 0:
                print("Empty sentence in line %i." % i)
                count_empty_sentences += 1
                continue
            s = s.split('\t')
            # index sentence words
            count_unk = 0
            indexed = []
            for w in s[attr_cols[0]].strip().split():
                word_id = dico.index(w, no_unk=False)
                if word_id < 4 + SPECIAL_WORDS and word_id != dico.unk_index:
                    logger.warning('Found unexpected special word "%s" (%i)!!' % (w, word_id))
                    continue
                indexed.append(word_id)
                if word_id == dico.unk_index:
                    unk_words[w] = unk_words.get(w, 0) + 1
                    count_unk += 1
            # index attributes
            sentence_attrs = []
            for attr, col in zip(attr_list, attr_cols[1:]):
                sentence_attrs.append(label2id[attr].get(s[col], None))
            # skip sentences with unknown attributes
            if any([x is None for x in sentence_attrs]):
                count_unknown_labels += 1
                continue
            # add sentence
            positions.append([len(sentences), len(sentences) + len(indexed)])
            sentences.extend(indexed)
            sentences.append(-1)
            attributes.append(sentence_attrs)

        f.close()

        print("Read %i sentences. %i were skipped because empty, and %i because contained unknown attributes."
              % (len(positions), count_empty_sentences, count_unknown_labels))

        # tensorize data
        positions = torch.LongTensor(positions)
        if len(dico) < 1 << 15:
            sentences = torch.ShortTensor(sentences)
        elif len(dico) < 1 << 31:
            sentences = torch.IntTensor(sentences)
        else:
            sentences = torch.LongTensor(sentences)
        attributes = torch.LongTensor(attributes)
        assert attributes.size() == (len(positions), len(attr_values))
        data = {
            'dico': dico,
            'attr_values': attr_values,
            'positions': positions,
            'sentences': sentences,
            'attributes': attributes,
            'unk_words': unk_words,
        }
        print("Saving the data to %s ..." % bin_path)
        torch.save(data, bin_path)

        return data
