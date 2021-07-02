#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import argparse
import numpy as np

# parse parameters
parser = argparse.ArgumentParser(description='Shuffle lines of parallel files')
parser.add_argument("--src_path", type=str, default="")
parser.add_argument("--tgt_path", type=str, default="")
params = parser.parse_args()

assert os.path.isfile(params.src_path)
assert os.path.isfile(params.tgt_path)


def read_lines(path):
    assert os.path.isfile(path)
    lines = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i % 1000000 == 0:
                print(i)
            lines.append(line)
    return lines


src_lines = read_lines(params.src_path)
tgt_lines = read_lines(params.tgt_path)

assert len(src_lines) == len(tgt_lines)
print("Read %i source and target lines." % len(src_lines))

# random generator with fixed seed
rng = np.random.RandomState(0)
permutation = rng.permutation(len(src_lines))

# write shuffled pairs of lines
with open(params.src_path + '.shuffled', 'w') as f:
    with open(params.tgt_path + '.shuffled', 'w') as g:
        for k, i in enumerate(permutation):
            if k % 1000000 == 0:
                print(k)
            f.write(src_lines[i])
            g.write(tgt_lines[i])
