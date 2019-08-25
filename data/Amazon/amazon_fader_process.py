"""
Script to process the Amazon Reviews data into a fader network style format where:

There are 2 files

1. All of the reviews (with nothing thrown away)
2. Metadata associated with each review Gender / Sentiment / Product Category

See: /prviate/home/subramas/Data/amazon_tmp/processed/style_transfer/fader/README

Requires:

1. --gender_folder    - A folder that contains files names.male and names.female
2. --user_fname       - Path to the file that contains the name of the user that wrote review
3. --review_fname     - Path to the file that contains the review text
4. --output           - Path to the folder to write the outputs
5. --categories_fname - Path to the file that contains all the product categories associated with the review
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

# List categories
books = book_cat_filter = set([
    'Books', 'Books & Comics',
    "Children's Books", "Children's eBooks",
    'Christian Books & Bibles', 'Comic Books', 'English Literature',
    'Kindle eBooks', 'Kindle Short Reads', 'Literature',
    'Literature & Fiction'
])
electronics = set([
    'Amazon Fire TV', 'Car & Vehicle Electronics', 'Car Electronics',
    'Cell Phones', 'Computer Mice',
    'Computer Speakers', 'Computer Workstations', 'Electrical & Electronics',
    'Electronics', 'Electronics & Audio',
    'Electronics & Gadgets', 'Fire TV', 'Gadgets, PCs & Consumer Electronics',
    'Graphics Tablets', 'Internet Phone',
    'Laptop & Netbook Computer Accessories', 'Mobile & Tablet',
    'Mobile Phones, Tablets & E-Readers', 'Mobiles',
    'No-Contract Cell Phones', 'Office Electronics', 'Phones', 'TV',
    'Unlocked Cell Phones', 'Tablets', 'Headphones'
])
movies = set([
    'Movies',
    'Movies & TV',
    'Movies & Video',
    'TV & Film'
])
clothing = set([
    'Clothing, Shoes & Jewelry',
    'Clothing',
    'Baby Clothing',
    'Clothing Accessories',
    'Fashion'
])
music = set([
    'CDs & Vinyl',
    'Music',
    'Digitial Music',
    "Children's Music",
    'World Music',
    'Electronic Music',
])

top_categories = [
    'Accessories',
    'Apps for Android',
    'Books',
    'CDs & Vinyl',
    'Cases',
    'Cell Phones & Accessories',
    'Clothing, Shoes & Jewelry',
    'Computers & Accessories',
    'Electronics',
    'Games',
    'Health & Personal Care',
    'Home & Kitchen',
    'Kindle Store',
    'Kindle eBooks',
    'Kitchen & Dining',
    'Literature & Fiction',
    'Men',
    'Movies',
    'Movies & TV',
    'N/A',
    'Pop',
    'Rock',
    'Romance',
    'Sports & Outdoors',
    'Women'
]

top_categories = {item: idx for idx, item in enumerate(top_categories)}

f_fader_review = open(os.path.join(args.output, 'fader.review'), 'w')
f_fader_attr = open(os.path.join(args.output, 'fader.attr'), 'w')

amazon_name_trigger_words = set([
    'Amazon', 'Kindle', 'N/A', 'Midwest', 'A', 'Anonymous',
    'J', 'Avid', 'Me', 'M', 'JJ', 'Unknown',
    'CJ', 'JP', 'AJ', 'JD'
])

for idx, (uname, review, category, score) in enumerate(zip(
    open(args.user_fname, 'r'),
    open(args.review_fname, 'r'),
    open(args.categories_fname, 'r'),
    open(args.scores_fname, 'r')
)):
    if idx % 100000 == 0:
        print('Finished %d lines' % (idx))

    #########################
    ####### (Gender) ########
    #########################

    uname = uname.strip()
    # Process Gender Annotation
    if uname == '' or not uname.split() or uname.split()[0] in amazon_name_trigger_words:
        gender_label = 2
    else:
        hash_object = hashlib.sha256(
            uname.lower().split()[0].replace(' ', '').encode()
        )
        hex_dig = hash_object.hexdigest()
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
    if category == '':
        parent_category_label = [0, 0, 0, 0, 0]
        subcategory_label = [0 for i in range(len(top_categories))]
    else:
        category = category.split('\t')
        parent_category_label = [0, 0, 0, 0, 0]
        subcategory_label = [0 for i in range(len(top_categories))]
        for item in category:
            # Parent Categories
            if item in books:
                parent_category_label[0] = 1

            if item in electronics:
                parent_category_label[1] = 1

            if item in movies:
                parent_category_label[2] = 1

            if item in clothing:
                parent_category_label[3] = 1

            if item in music:
                parent_category_label[4] = 1

            if item in top_categories:
                subcategory_label[top_categories[item]] = 1

    # Write results
    f_fader_review.write(review)
    f_fader_attr.write(
        str(gender_label) + '\t' +
        str(score_label) + '\t' +
        str(binary_score) + '\t' +
        '\t'.join([str(x) for x in parent_category_label]) + '\t' +
        '\t'.join([str(x) for x in subcategory_label]) +
        '\n'
    )

f_fader_review.close()
f_fader_attr.close()
