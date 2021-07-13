#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# -*- coding: utf-8 -*-
import os
import sys

from src.attributes import read_attr_values
from src.logger import create_logger
from src.data.dictionary import Dictionary


if __name__ == '__main__':

    logger = create_logger(None)

    # read arguments
    voc_path = sys.argv[1]
    lbl_path = sys.argv[2]
    attr_list = sys.argv[3].split(',')
    attr_cols = sys.argv[4].split(',')  # column ID of text followed by attributes in the same order
    txt_path = sys.argv[5]
    bin_path = sys.argv[5] + '.pth'

    # arguments check
    assert os.path.isfile(voc_path)
    assert os.path.isfile(lbl_path)
    assert os.path.isfile(txt_path)
    assert all(attr_list) and len(attr_list) == len(set(attr_list)) >= 1
    assert all(attr_cols) and len(attr_cols) == len(set(attr_cols)) == len(attr_list) + 1
    # sort attributes
    review_col = attr_cols[0]
    sorted_attr = sorted(list(zip(attr_list, attr_cols[1:])))
    attr_list = [x[0] for x in sorted_attr]
    attr_cols = [x[1] for x in sorted_attr]
    attr_cols.insert(0, review_col)
    logger.info(attr_list)
    logger.info(attr_cols)
    assert attr_list == sorted(attr_list)

    # read vocabulary
    dico = Dictionary.read_vocab(voc_path)

    # read attribute labels
    attr_values = read_attr_values(lbl_path)
    print(sorted(attr_values.keys()), attr_list)
    assert sorted(attr_values.keys()) == attr_list

    # index and export data
    attr_cols = [int(x) for x in attr_cols]
    data = Dictionary.index_data(txt_path, bin_path, dico, attr_list, attr_cols, attr_values)
    logger.info("%i words (%i unique) in %i sentences." % (
        len(data['sentences']) - len(data['positions']),
        len(data['dico']),
        len(data['positions'])
    ))

    # print unknown words
    if len(data['unk_words']) > 0:
        logger.info("%i unknown words (%i unique), covering %.2f%% of the data." % (
            sum(data['unk_words'].values()),
            len(data['unk_words']),
            sum(data['unk_words'].values()) * 100. / (len(data['sentences']) - len(data['positions']))
        ))
        if len(data['unk_words']) < 30:
            for w, c in sorted(data['unk_words'].items(), key=lambda x: x[1])[::-1]:
                logger.info("%s: %i" % (w, c))
    else:
        logger.info("0 unknown word.")
