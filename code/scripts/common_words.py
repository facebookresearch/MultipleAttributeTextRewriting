#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys
from collections import OrderedDict

path1 = sys.argv[1]
path2 = sys.argv[2]

assert os.path.isfile(path1)
assert os.path.isfile(path2)


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


# read vocabulary
dico1 = read_vocab(path1)
dico2 = read_vocab(path2)

# most frequent words
if len(sys.argv) >= 4 and int(sys.argv[3]) > 0:
    dico1 = dict(dico1.items()[:int(sys.argv[3])])
    dico2 = dict(dico2.items()[:int(sys.argv[3])])

# unique words
words1 = set(dico1.keys())
words2 = set(dico2.keys())
common_words = words1 & words2

print("===== Unique words =====")
print("Read %i and %i words." % (len(words1), len(words2)))
print("Found %i common words." % len(words1 & words2))
print("%.8f%% of the words in 1 are in 2." % (100. * len(words1 & words2) / len(words1)))
print("%.8f%% of the words in 2 are in 1." % (100. * len(words1 & words2) / len(words2)))
print("")
print("===== Total words =====")
print("Read %i and %i words." % (sum([count for word, count in dico1.items()]), sum([count for word, count in dico2.items()])))
print("%.8f%% of the words in 1 are in 2." % (100. * sum([dico1[word] for word in common_words]) / sum([count for word, count in dico1.items()])))
print("%.8f%% of the words in 2 are in 1." % (100. * sum([dico2[word] for word in common_words]) / sum([count for word, count in dico2.items()])))
