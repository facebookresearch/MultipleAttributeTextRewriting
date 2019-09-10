#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


CODES='40000 60000 80000'
THREADS=24

DATA_PATH=../dataset/yelp
NAME_PATH=../names
PROC_DATA_PATH=$DATA_PATH/processed
FINAL_DATA_PATH=$PROC_DATA_PATH/style_transfer

BPE=../fastBPE/fast
FASTTEXT=../fastTextInstall/bin/fasttext

MOSES_PATH=../mosesdecoder
TOKENIZER=$MOSES_PATH/scripts/tokenizer/tokenizer.perl
NORMALIZER=$MOSES_PATH/scripts/tokenizer/normalize-punctuation.perl
LOWERCASER=$MOSES_PATH/scripts/tokenizer/lowercase.perl

LID_PATH=../models/lid.176.bin
CAT_CLF_PATH=../models/yelp_categories.model.bin

mkdir -p $DATA_PATH
mkdir -p $PROC_DATA_PATH
mkdir -p $FINAL_DATA_PATH

# Extract / Tokenize
echo "#############################"
echo "Extracting Reviews from JSON"
echo "#############################"

python extract_review.py $DATA_PATH/review.json text s | $NORMALIZER -l en > $PROC_DATA_PATH/reviews.txt

echo "#############################"
echo "Extracting Scores from JSON"
echo "#############################"

python extract_review.py $DATA_PATH/review.json stars f > $PROC_DATA_PATH/scores.txt

echo "#############################"
echo "Tokenizing Reviews"
echo "#############################"

$TOKENIZER -l en -no-escape -threads $THREADS < $PROC_DATA_PATH/reviews.txt > $PROC_DATA_PATH/reviews.tok

echo "#############################"
echo "Score Distribution"
echo "#############################"

cat $PROC_DATA_PATH/scores.txt | sort | uniq -c

# Classify reviews by language / Filter English reviews
echo "#############################"
echo "Filtering English Reviews"
echo "#############################"

$FASTTEXT predict $LID_PATH $PROC_DATA_PATH/reviews.txt > $PROC_DATA_PATH/langs.txt
python filter.py $PROC_DATA_PATH/reviews.tok $PROC_DATA_PATH/langs.txt __label__en | $LOWERCASER > $PROC_DATA_PATH/reviews.tok.low.en

# Get Yelp data with annotations from meta-data about username and business name
echo "#############################"
echo "Fetching Meta-data"
echo "#############################"

python merge_yelp.py -bm $DATA_PATH/business.json -um $DATA_PATH/user.json -r $DATA_PATH/review.json -f $PROC_DATA_PATH

# Filter rest of the meta-data for the english language only
echo "#############################"
echo "Filtering English content from meta-data"
echo "#############################"

python filter.py $PROC_DATA_PATH/scores.txt $PROC_DATA_PATH/langs.txt __label__en > $PROC_DATA_PATH/scores.txt.en
python filter.py $PROC_DATA_PATH/categories.txt $PROC_DATA_PATH/langs.txt __label__en > $PROC_DATA_PATH/categories.txt.en
python filter.py $PROC_DATA_PATH/reviewer_name.txt $PROC_DATA_PATH/langs.txt __label__en > $PROC_DATA_PATH/reviewer_name.txt.en

echo "#############################"
echo "Creating Processed dataset"
echo "#############################"

python yelp_fader_process.py \
    -gf $NAME_PATH \
    -rf $PROC_DATA_PATH/reviews.tok.low.en \
    -sf $PROC_DATA_PATH/scores.txt.en \
    -uf $PROC_DATA_PATH/reviewer_name.txt.en \
    -cf $PROC_DATA_PATH/categories.txt.en \
    -o  $FINAL_DATA_PATH

rm $PROC_DATA_PATH/categories.txt
rm $PROC_DATA_PATH/categories.txt.en
rm $PROC_DATA_PATH/langs.txt
rm $PROC_DATA_PATH/reviewer_name.txt
rm $PROC_DATA_PATH/reviewer_name.txt.en
rm $PROC_DATA_PATH/reviews.tok
rm $PROC_DATA_PATH/reviews.tok.low.en
rm $PROC_DATA_PATH/reviews.txt
rm $PROC_DATA_PATH/scores.txt
rm $PROC_DATA_PATH/scores.txt.en

echo "#############################"
echo "Learning and applying BPE"
echo "#############################"

for CODE in $CODES; do
    $BPE learnbpe $CODE $FINAL_DATA_PATH/fader.review > $FINAL_DATA_PATH/codes.$CODE
    $BPE applybpe $FINAL_DATA_PATH/fader.review.$CODE $FINAL_DATA_PATH/fader.review $FINAL_DATA_PATH/codes.$CODE
    $BPE getvocab $FINAL_DATA_PATH/fader.review.$CODE > $FINAL_DATA_PATH/vocab.$CODE
done

echo "#############################"
echo "Merging data and attributes"
echo "#############################"

for CODE in $CODES; do
    paste -d "\t" $FINAL_DATA_PATH/fader.review.$CODE $FINAL_DATA_PATH/fader.attr > $FINAL_DATA_PATH/fader.$CODE
done

get_seeded_random()
{
    seed="$1"
    openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
        </dev/zero 2>/dev/null
}

echo "#############################"
echo "Splitting train/val/test data"
echo "#############################"

NLINES=`wc -l $FINAL_DATA_PATH/fader.$CODE | awk -F " " '{print $1}'`
NTRAIN=$((NLINES - 20000))
NVAL=$((NTRAIN + 10000))
for CODE in $CODES; do
    shuf --random-source=<(get_seeded_random 42) $FINAL_DATA_PATH/fader.$CODE | head -$NTRAIN > $FINAL_DATA_PATH/train.fader.$CODE
    shuf --random-source=<(get_seeded_random 42) $FINAL_DATA_PATH/fader.$CODE | head -$NVAL | tail -10000 > $FINAL_DATA_PATH/valid.fader.$CODE
    shuf --random-source=<(get_seeded_random 42) $FINAL_DATA_PATH/fader.$CODE | tail -10000 > $FINAL_DATA_PATH/test.fader.$CODE
done

echo "#############################"
echo "Generating categorical labels for restaurant type"
echo "#############################"

# Generate labels
for NAME in valid test; do
    $FASTTEXT predict $CAT_CLF_PATH \
    <(awk -F "\t" '{print $1}' $FINAL_DATA_PATH/$NAME.fader.60000 | sed -r 's/(@@ )|(@@ ?$)//g') \
    | sed 's/^__label__//' \
    > $FINAL_DATA_PATH/$NAME.categories
done

mkdir -p $FINAL_DATA_PATH/split
cut -f1 $FINAL_DATA_PATH/train.fader.80000 | sed -r 's/(@@ )|(@@ ?$)//g' | split -l 1000000 - $FINAL_DATA_PATH/split/
for FILENAME in aa ab ac ad ae af; do
    $FASTTEXT predict $CAT_CLF_PATH $FINAL_DATA_PATH/split/$FILENAME \
    | sed 's/^__label__//' > $FINAL_DATA_PATH/split/$FILENAME.categories &
done

wait

cat $FINAL_DATA_PATH/split/aa.categories $FINAL_DATA_PATH/split/ab.categories $FINAL_DATA_PATH/split/ac.categories $FINAL_DATA_PATH/split/ad.categories $FINAL_DATA_PATH/split/ae.categories $FINAL_DATA_PATH/split/af.categories > $FINAL_DATA_PATH/train.categories

# Add labels
for NAME in train valid test; do
    for CODE in $CODES; do
        paste -d '\t' $FINAL_DATA_PATH/$NAME.fader.$CODE $FINAL_DATA_PATH/$NAME.categories > $FINAL_DATA_PATH/$NAME.fader.with_cat.$CODE
    done
done

echo "#############################"
echo "Removing some label attributes"
echo "#############################"

for i in gender,2 sentiment,3 binary_sentiment,4 restaurant,5 asian,6 american,7 mexican,8 bar,9 dessert,10 type,34; do
    IFS=","; set -- $i;
    cut -f$2 $FINAL_DATA_PATH/train.fader.with_cat.80000 \
    | awk '{a[$1]+=1}END{for(k in a){print a[k] " " k}}' \
    | awk '{a=$1; $1=$2; $2=a}1' \
    | sed "1 i$1" \
    | sed ':a;N;$!ba;s/\n/|||/g'
done | tee $FINAL_DATA_PATH/labels

# Remove unused labels like reviews that aren't about restaurants.
python ../remove_unused_labels.py -lf $FINAL_DATA_PATH/labels --gender > $FINAL_DATA_PATH/labels.proc

echo "#############################"
echo "Removing human written test examples from automatic train/val/test sets"
echo "#############################"

# Remove human test set examples from train/valid/test.
for CODE in $CODES; do
    python ../remove_human_test.py \
        -fp $FINAL_DATA_PATH/train.fader.with_cat.$CODE \
        -ref ../human_test/yelp/hashed/all.hash \
        -out $FINAL_DATA_PATH/train.fader.with_cat.proc.$CODE
    python ../remove_human_test.py \
        -fp $FINAL_DATA_PATH/valid.fader.with_cat.$CODE \
        -ref ../human_test/yelp/hashed/all.hash \
        -out $FINAL_DATA_PATH/valid.fader.with_cat.proc.$CODE
    python ../remove_human_test.py \
        -fp $FINAL_DATA_PATH/test.fader.with_cat.$CODE \
        -ref ../human_test/yelp/hashed/all.hash \
        -out $FINAL_DATA_PATH/test.fader.with_cat.proc.$CODE
done

echo "#############################"
echo "Creating human test set from hashes"
echo "#############################"

python ../create_human_test_set.py \
    -fpath ../human_test/yelp/hashed \
    -ref $FINAL_DATA_PATH/fader.review \
    -opath ../human_test/yelp/proc