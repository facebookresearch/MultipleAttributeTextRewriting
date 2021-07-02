# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

LABELS='gender,sentiment,binary_sentiment,restaurant,asian,american,mexican,bar,dessert,type'
COLUMN_IDS='0,1,2,3,4,5,6,7,8,9,33'

for CODE in 40000 60000 80000; do
  for NAME in train valid test; do
    python preprocess.py ../data/dataset/yelp/processed/style_transfer/"vocab.$CODE" ../data/dataset/yelp/processed/style_transfer/labels.proc "$LABELS" "$COLUMN_IDS" ../data/dataset/yelp/processed/style_transfer/"$NAME.fader.with_cat.proc.$CODE"
  done
done
