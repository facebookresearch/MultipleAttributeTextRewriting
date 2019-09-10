# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Script to create the human test set where the input sentence is hashed.

Requires:

1. -fpath - Path to the folder that contains .hash files that need to be modified.
2. -ref   - Path to the file that contains all reviews in the dataset.
3. -out   - Path to the file that will contain the modified file.
Output:
A file that contains removed examples.
"""

import argparse
import hashlib
import os
from aes_utils import AESCipher

parser = argparse.ArgumentParser()

parser.add_argument(
    '-fpath', '--folder_path', required=True,
    help="Path to the folder that contains hashed input sentences"
)
parser.add_argument(
    '-ref', '--ref_file', required=True,
    help="Path to the file that contains all of the reviews"
)
parser.add_argument(
    '-opath', '--output_path', required=True,
    help="Path to the output folder where files are to be written"
)
args = parser.parse_args()

assert os.path.exists(args.folder_path)
assert os.path.exists(args.output_path)
assert os.path.exists(args.ref_file)

aes = AESCipher()

# Collect all required hashes
required_hashes = set()
for fname in os.listdir(args.folder_path):
    if fname == 'all.hash':
        continue
    else:
        fname = os.path.join(args.folder_path, fname)
        with open(fname, 'r') as f:
            for line in f:
                hashed_input = line.strip().split('\t')[0]
                required_hashes.add(hashed_input)

print('Searching for %d hashes' % (len(required_hashes)))

hash_sentence_map = {}
with open(args.ref_file, 'r') as f:
    for line in f:
        hash_object = hashlib.sha256(line.strip().replace(' ', '').encode())
        hex_dig = hash_object.hexdigest()
        if hex_dig in required_hashes:
            hash_sentence_map[hex_dig] = line.strip()

print('Found %d/%d hashes' % (len(hash_sentence_map), len(required_hashes)))
for fname in os.listdir(args.folder_path):
    if fname == 'all.hash':
        continue
    else:
        out_fname = os.path.join(
            args.output_path, fname.replace('.hash', '.tsv')
        )
        fname = os.path.join(args.folder_path, fname)
        with open(fname, 'r') as f, open(out_fname, 'w') as fout:
            for line in f:
                line = line.strip().split('\t')
                if line[0] in hash_sentence_map:
                    inp = hash_sentence_map[line[0]]
                    inp_hash = hashlib.md5(
                        inp.strip().replace(' ', '').encode()
                    ).hexdigest()
                    decrypted_out = aes.decrypt(line[1], inp_hash)
                    fout.write(
                        inp + '\t' + decrypted_out + '\n'
                    )
