# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Script to remove examples that appear in the human rewrite set from train/valid/test.

Requires:

1. -fp    - Path to the file that needs to be modified.
2. -ref   - Path to the file that contains hashes of the examples that occur in the human test set.
3. -out   - Path to the file that will contain the modified file.
Output:
A file that contains removed examples.
"""

import argparse
import hashlib
parser = argparse.ArgumentParser()

parser.add_argument(
    '-fp', '--inp_file', required=True,
    help="Path to the file that needs to be modified"
)
parser.add_argument(
    '-ref', '--ref_file', required=True,
    help="Path to the file that contains the hash of the examples to be removed"
)
parser.add_argument(
    '-out', '--out_file', required=True,
    help="Path to the output file"
)
args = parser.parse_args()

examples_removed = 0

ref_hashes = set([line.strip() for line in open(args.ref_file, 'r')])

with open(args.inp_file, 'r') as f, open(args.out_file, 'w') as fout:
    for idx, line in enumerate(f):
        line = line.strip().split('\t')
        hash_object = hashlib.sha256(
            line[0].strip().lower().replace('@@ ', '').replace(' ', '').encode()
        )
        hex_dig = hash_object.hexdigest()
        if hex_dig in ref_hashes:
            examples_removed += 1
        else:
            fout.write('\t'.join(line) + '\n')
