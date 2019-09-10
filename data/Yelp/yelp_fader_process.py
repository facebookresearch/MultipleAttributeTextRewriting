# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Script to process the Yelp Reviews data into a fader network style format:

Requires:

1. --gender_folder    - A folder that contains files names.male and names.female
2. --user_fname       - Path to the file that contains the name of the user that wrote review
3. --review_fname     - Path to the file that contains the review text
4. --output           - Path to the folder to write the outputs
5. --categories_fname - Path to the file that contains all the business categories associated with the review
6. --scores_fname     - Path to the file that contains the star rating of the review

Output:
1. Review file (fader.review) - Should copy the review_fname as is
2. Attr file   (fader.attr)   - Tab separated attributes
ex: 1       2       0       0       0,0,0,0,0       1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
"""
import os
import argparse
import hashlib

parser = argparse.ArgumentParser()

parser.add_argument(
    '-gf', '--gender_folder', required=True,
    help="Path to the folder with names in each gender"
)
parser.add_argument(
    '-uf', '--user_fname', required=True,
    help="Filename with usernames"
)
parser.add_argument(
    '-rf', '--review_fname', required=True,
    help="Filename with reviews"
)
parser.add_argument(
    '-o', '--output', required=True,
    help="Output folder to write results"
)
parser.add_argument(
    '-cf', '--categories_fname', required=True,
    help="Path to the business categories"
)
parser.add_argument(
    '-sf', '--scores_fname', required=True,
    help="Path to the file with scores"
)
args = parser.parse_args()

# Gender Sentiment Descriptive Personal RestaurantOrNot ParentCategories SubCategories

male_names = set(
    [line.strip().lower() for line in open(os.path.join(args.gender_folder, 'names.male'), 'r')]
)
female_names = set(
    [line.strip().lower() for line in open(os.path.join(args.gender_folder, 'names.female'), 'r')]
)

# Make sure intersection is null.
assert not male_names & female_names

assert os.path.isfile(args.user_fname)
assert os.path.isfile(args.review_fname)
assert os.path.isfile(args.categories_fname)
assert os.path.isfile(args.scores_fname)


# Parent categories list
asian = set([
    'Japanese', 'Thai', 'Ramen', 'Sushi', 'Sushi Bar',
    'Chinese', 'Asian Fusion', 'Vietnamese', 'Korean',
    'Noodles', 'Dim Sum', 'Cantonese', 'Filipino', 'Taiwanese'
])
american = set([
    'American (New)', 'American (Traditional)',
    'Canadian (New)', 'Southern'
])
mexican = set([
    'New Mexican Cuisine', 'Mexican', 'Tacos',
    'Tex-Mex', 'Tapas Bars', 'Latin American'
])
bar = set([
    'Brasseries', 'Nightlife', 'Bars', 'Pubs', 'Wine Bars',
    'Sports Bars', 'Beer', 'Cocktail Bars'
])
dessert = set([
    'Desserts', 'Bakeries', 'Ice Cream & Frozen Yogurt',
    'Juice Bars & Smoothies', 'Donuts', 'Cupcakes', 'Chocolatiers & Shops'
])

top_categories = [
    'Nightlife',
    'Bars',
    'American (New)',
    'American (Traditional)',
    'Breakfast & Brunch',
    'Sandwiches',
    'Mexican',
    'Pizza',
    'Italian',
    'Burgers',
    'Seafood',
    'Coffee & Tea',
    'Japanese',
    'Chinese',
    'Sushi Bars',
    'Desserts',
    'Steakhouses',
    'Asian Fusion',
    'Salad',
    'Fast Food',
    'Cafes',
    'Event Planning & Services',
    'Bakeries'
]

top_categories = {item: idx for idx, item in enumerate(top_categories)}


f_fader_review = open(os.path.join(args.output, 'fader.review'), 'w')
f_fader_attr = open(os.path.join(args.output, 'fader.attr'), 'w')

for idx, (uname, review, category, score) in enumerate(zip(
    open(args.user_fname, 'r'),
    open(args.review_fname, 'r'),
    open(args.categories_fname, 'r'),
    open(args.scores_fname, 'r')
)):

    #########################
    ####### (Gender) ########
    #########################

    if idx % 100000 == 0:
        print('Finished %d lines' % (idx))
    uname = uname.strip()
    hash_object = hashlib.sha256(uname.lower().replace(' ', '').encode())
    hex_dig = hash_object.hexdigest()
    if uname == '':
        gender_label = 2
    else:
        if hex_dig in male_names:
            gender_label = 0
        elif hex_dig in female_names:
            gender_label = 1
        else:
            gender_label = 2

    #########################
    ######## (Score) ########
    #########################

    score_label = int(float(score.strip()))
    assert score_label in [1, 2, 3, 4, 5]
    if score_label in [4, 5]:
        binary_score = 1
    elif score_label in [1, 2]:
        binary_score = -1
    else:
        binary_score = 0

    #########################
    ###### (Category) #######
    #########################

    category = category.strip()
    if category == '' or category == 'N/A':
        parent_category_label = [0, 0, 0, 0, 0]
        subcategory_label = [0 for i in range(len(top_categories))]
    else:
        category = category.split('\t')

        if 'Restaurants' not in category and 'Food' not in category:
            restaurantornot_category_label = 0
        else:
            restaurantornot_category_label = 1

        parent_category_label = [0, 0, 0, 0, 0]
        subcategory_label = [0 for i in range(len(top_categories))]
        for item in category:
            # Parent Categories
            if item in asian:
                parent_category_label[0] = 1

            if item in american:
                parent_category_label[1] = 1

            if item in mexican:
                parent_category_label[2] = 1

            if item in bar:
                parent_category_label[3] = 1

            if item in dessert:
                parent_category_label[4] = 1

            if item in top_categories:
                subcategory_label[top_categories[item]] = 1

    # Write results
    f_fader_review.write(review)
    f_fader_attr.write(
        str(gender_label) + '\t' +
        str(score_label) + '\t' +
        str(binary_score) + '\t' +
        str(restaurantornot_category_label) + '\t' +
        '\t'.join([str(x) for x in parent_category_label]) + '\t' +
        '\t'.join([str(x) for x in subcategory_label]) +
        '\n'
    )

f_fader_review.close()
f_fader_attr.close()
