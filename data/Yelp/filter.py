# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys


assert len(sys.argv) >= 3
filepath1 = sys.argv[1]
filepath2 = sys.argv[2]
k = sys.argv[3]
assert os.path.isfile(filepath1)
assert os.path.isfile(filepath2)

f1 = open(filepath1, 'r')
f2 = open(filepath2, 'r')
for line1, line2 in zip(f1, f2):
    if line2.rstrip() == k:
        print(line1.rstrip())
f1.close()
f2.close()
