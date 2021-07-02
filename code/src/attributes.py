# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import torch


logger = getLogger()


IGNORED_LABELS = ['null', 'unknown']
IGNORED_ATTR_LABELS = [('gender', '2')]


def read_attr_values(path):

    attr_values = {}

    # read all attribute labels
    with open(path, 'r') as f:
        for line in f:
            split = line.rstrip().split('|||')
            assert len(split) >= 2
            attr = split[0]
            attr_values[attr] = []
            # last_count = 1e12
            for x in split[1:]:
                label, count = x.split()
                assert '__' not in attr and '__' not in label
                if label.lower() in IGNORED_LABELS:
                    continue
                if (attr.lower(), label.lower()) in IGNORED_ATTR_LABELS:
                    continue
                attr_values[attr].append(label)
                # # check attributes were sorted
                # assert int(count) <= last_count
                # last_count = int(count)

    logger.info("Found %i attribute categories:" % len(attr_values))
    for attr, labels in attr_values.items():
        logger.info("\t%s: %s" % (attr, ", ".join(labels)))

    return attr_values


def gen_rand_attr(bs, attributes, attr_values):
    """
    Generate a random set of attribute labels.
    """
    attr = torch.LongTensor(bs, len(attributes))
    offset = 0
    for i, a in enumerate(attributes):
        n = len(attr_values[a])
        attr[:, i] = torch.LongTensor(bs).random_(n) + offset
        offset += n
    return attr
