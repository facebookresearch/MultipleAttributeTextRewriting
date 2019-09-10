# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Script to pre-process the Amazon Reviews to aggregate reivews and metadata together:

Requires:
1. --business_meta - A json file that contains all the metadata for for businesses
2. --user_meta     - A json file that contains all the metadata about a user (name etc.)
3. --review        - A json file that contains the review content
4. --folder        - Path to the folder to write outputs

Output:
1. Review file    (reviews.txt)       - Extracted review content from JSON
2. Reviewer Names (reviewer_name.txt) - Extracted reviewer name
3. Score          (scores.txt)         - Star rating of the review
4. Categories     (categories.txt)    - Product categories for each review
"""
import os
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    '-bm', '--business_meta', required=True, help="Business metadata"
)
parser.add_argument(
    '-um', '--user_meta', required=True, help="User info metadata"
)
parser.add_argument(
    '-r', '--review', required=True, help="Review file"
)
parser.add_argument(
    '-f', '--folder', required=True, help="Folder to write outputs"
)
args = parser.parse_args()

"""
Attributes to collect

1. Review content
2. Reviewer name
3. Review score
4. Business categories
"""

# First Read the meta-data and collect everything into a dictionary.

b_meta_dict = {}
b_meta_file = open(args.business_meta, 'r')
print('Processing Business metadata file ...')
for idx, line in enumerate(b_meta_file):
    if idx % 100000 == 0:
        print('Finished %d lines ' % (idx))
    parsed_line = json.loads(line)
    if 'categories' in parsed_line:
        categories = parsed_line['categories']
        b_meta_dict[parsed_line['business_id']] = categories

u_meta_dict = {}
u_meta_file = open(args.user_meta, 'r')
print('Processing User metadata file ...')
for idx, line in enumerate(u_meta_file):
    if idx % 100000 == 0:
        print('Finished %d lines ' % (idx))
    parsed_line = json.loads(line)
    if 'name' in parsed_line:
        name = parsed_line['name']
        u_meta_dict[parsed_line['user_id']] = name

f_rev = open(os.path.join(args.folder, 'reviews.txt'), 'w')
f_rev_name = open(os.path.join(args.folder, 'reviewer_name.txt'), 'w')
f_score = open(os.path.join(args.folder, 'scores.txt'), 'w')
f_cat = open(os.path.join(args.folder, 'categories.txt'), 'w')

review_file = open(args.review, 'r')
print('Processing review file ...')
for idx, line in enumerate(review_file):
    if idx % 100000 == 0:
        print('Finished %d lines ' % (idx))
    parsed_line = json.loads(line)
    uid = parsed_line['user_id']
    bid = parsed_line['business_id']
    review = parsed_line['text'].replace('\n', '')
    score = str(parsed_line['stars'])
    if uid in u_meta_dict:
        uname = u_meta_dict[uid]
    else:
        uname = 'N/A'
    if bid in b_meta_dict and b_meta_dict[bid] is not None:
        bcat = '\t'.join(b_meta_dict[bid].split(', '))
    else:
        bcat = 'N/A'
    f_rev.write(review.strip() + '\n')
    f_rev_name.write(uname.strip() + '\n')
    f_score.write(str(score) + '\n')
    f_cat.write(bcat + '\n')

f_rev.close()
f_rev_name.close()
f_score.close()
f_cat.close()
