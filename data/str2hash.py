#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import hashlib
import sys

if __name__ == '__main__':
    for line in sys.stdin:
        hash_object = hashlib.sha256(line.strip().replace(' ', '').encode())
        hex_dig = hash_object.hexdigest()
        print(hex_dig)
