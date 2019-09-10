# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Script to remove unused label categories from the label meta-data file:

Requires:

1. -lf    - A file that contains label meta-data information that was automatically generated.

Output:
STDOUT - The modified file with certain attribute labels removed
"""
import argparse
parser = argparse.ArgumentParser()

parser.add_argument(
    '-lf', '--label_file', required=True,
    help="Path to the file that contains label meta-data"
)
parser.add_argument(
    '-g', '--gender', action="store_true",
    help="Whether to remove examples where gender is unknown"
)
args = parser.parse_args()

lines = [line.strip().split('|||') for line in open(args.label_file, 'r')]

modified_lines = []

for line in lines:
    modified_line = []
    if line[0] == 'gender' and args.gender:
        modified_line.append(line[0])
        for attr in line[1:]:
            if attr.split()[0] != '2':
                modified_line.append(attr)
    elif line[0] == 'sentiment':
        modified_line.append(line[0])
        for attr in line[1:]:
            if attr.split()[0] != '3':
                modified_line.append(attr)
    elif line[0] == 'binary_sentiment':
        modified_line.append(line[0])
        for attr in line[1:]:
            if attr.split()[0] != '0':
                modified_line.append(attr)
    elif line[0] == 'restaurant':
        modified_line.append(line[0])
        for attr in line[1:]:
            if attr.split()[0] != '0':
                modified_line.append(attr)
    else:
        modified_line = line
    modified_lines.append(modified_line)

print('\n'.join(['|||'.join(line) for line in modified_lines]))
