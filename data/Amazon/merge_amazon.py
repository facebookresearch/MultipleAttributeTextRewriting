# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Script to pre-process the Amazon Reviews to aggregate reivews and metadata together:

Requires:
1. --meta   - A json file that contains all the metadata for reviews
2. --review - A json file that contains the raw reviews and product scores
3. --folder - Path to write the resulting files

Output:
1. Review file    (reviews.txt)       - Extracted review content from JSON
2. Reviewer Names (reviewer_name.txt) - Extracted reviewer name
3. Summary        (summary.txt)       - Summary of the review
4. Helpful        (helpful.txt)       - Helpfulness rating of the review
5. Score          (scores.txt)         - Star rating of the review
6. Categories     (categories.txt)    - Product categories for each review
"""
import os
import ast
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    '-m', '--meta', default='metadata.json', help="Product metadata"
)
parser.add_argument(
    '-r', '--review', default='aggressive_dedup.json', help="Review file"
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
4. Product categories
"""

# First Read the meta-data and collect everything into a dictionary.

meta_dict = {}
meta_file = open(args.meta, 'r')
print('Processing metadata file ...')
for idx, line in enumerate(meta_file):
    if idx % 100000 == 0:
        print('Finished %d lines ' % (idx))
    parsed_line = ast.literal_eval(line)
    if 'categories' in parsed_line:
        categories = parsed_line['categories']
        meta_dict[parsed_line['asin']] = sum(categories, [])

f_rev = open(os.path.join(args.folder, 'reviews.txt'), 'w')
f_rev_name = open(os.path.join(args.folder, 'reviewer_name.txt'), 'w')
f_summary = open(os.path.join(args.folder, 'summary.txt'), 'w')
f_help = open(os.path.join(args.folder, 'helpful.txt'), 'w')
f_score = open(os.path.join(args.folder, 'scores.txt'), 'w')
f_cat = open(os.path.join(args.folder, 'categories.txt'), 'w')

review_file = open(args.review, 'r')
print('Processing review file ...')
for idx, line in enumerate(review_file):
    if idx % 100000 == 0:
        print('Finished %d lines ' % (idx))
    parsed_line = json.loads(line)
    asin = parsed_line['asin']
    review = parsed_line['reviewText']
    if 'reviewerName' in parsed_line:
        reviewer_name = parsed_line['reviewerName'].replace('\n', '')
    else:
        reviewer_name = 'N/A'

    if 'summary' in parsed_line:
        summary = parsed_line['summary']
    else:
        summary = 'N/A'
    helpful = parsed_line['helpful']
    score = parsed_line['overall']
    f_rev.write(review.strip() + '\n')
    f_rev_name.write(reviewer_name.strip() + '\n')
    f_summary.write(summary.strip() + '\n')
    helpful = str(helpful[0]) + '/' + str(helpful[1])
    f_help.write(helpful + '\n')
    f_score.write(str(score) + '\n')
    if asin in meta_dict:
        prod_cat = '\t'.join(meta_dict[asin])
    else:
        prod_cat = 'N/A'
    f_cat.write(prod_cat + '\n')

f_rev.close()
f_rev_name.close()
f_summary.close()
f_help.close()
f_score.close()
f_cat.close()
