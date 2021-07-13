#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys

path1 = sys.argv[1]
path2 = sys.argv[2]

assert os.path.isfile(path1)
assert os.path.isfile(path2)


def read_lines(path):
    assert os.path.isfile(path)
    with open(path, 'r') as f:
        lines = [line for line in f]
    return lines


# read lines
lines1 = read_lines(path1)
lines2 = read_lines(path2)

# max lines
if len(sys.argv) >= 4 and int(sys.argv[3]) > 0:
    lines1 = lines1[:int(sys.argv[3])]
    lines2 = lines2[:int(sys.argv[3])]

lines1 = set(lines1)
lines2 = set(lines2)

print("Read %i and %i lines." % (len(lines1), len(lines2)))
print("Found %i common lines." % len(lines1 & lines2))
print("%.8f%% of the lines in 1 are in 2." % (100. * len(lines1 & lines2) / len(lines1)))
print("%.8f%% of the lines in 2 are in 1." % (100. * len(lines1 & lines2) / len(lines2)))
