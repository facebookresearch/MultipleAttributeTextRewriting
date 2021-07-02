# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python
import os
import sys
from collections import OrderedDict

voc_path = sys.argv[1]
txt_path = sys.argv[2]

assert os.path.isfile(voc_path)
assert os.path.isfile(txt_path)


def read_vocab(path):
    assert os.path.isfile(path)
    dico = OrderedDict()
    with open(path, 'r') as f:
        for line in f:
            line = line.rstrip().split()
            assert len(line) == 2
            dico[line[0]] = int(line[1])
    assert all(x > 0 for x in dico.values())
    return dico


def read_text(path):
    assert os.path.isfile(path)
    words = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.rstrip().split()
            for word in line:
                words[word] = words.get(word, 0) + 1
    return words


# read vocab and text file
vocab = read_vocab(voc_path)
words = read_text(txt_path)
# print("Read %i words in the vocabulary file (%i unique)."
#       % (sum(vocab.values()), len(vocab)))
# print("Read %i words in the text file (%i unique)."
#       % (sum(words.values()), len(words)))

for word, count in sorted(words.items(), key=lambda x: -x[1]):
    if word not in vocab:
        assert len(word) == 1 or len(word) == 3 and word.endswith('@@'), word
        print("{: >3} was not found in the vocabulary ({} occurrences)".format(word, count))
