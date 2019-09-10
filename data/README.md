# Dataset pre-processing.

This Readme helps setup the FYelp and Amazon datasets preseted in our ICLR'19 paper [Multiple-Attribute Text Style Transfer
](https://arxiv.org/abs/1811.00552).

## Dependencies

* Python 3
* [fastText](https://github.com/facebookresearch/fastText)
* [fastBPE](https://github.com/glample/fastBPE) 
* [Moses](https://github.com/moses-smt/mosesdecoder) (Only to pre-process the data, no installation required)
* [pycrypto](https://pypi.org/project/pycrypto/)

Make sure you have at least 500GB of disk space for the Amazon and Yelp datasets combined.

## Setup Dependencies

First, make sure you're in the `data` folder of the main repository (where this README is located).

### 1. fastText

```
git clone https://github.com/facebookresearch/fastText.git
cd fastText
mkdir build && cd build && cmake .. -DCMAKE_INSTALL_PREFIX=../../fastTextInstall
make -j 12 && make install
cd ../..
```

This should create the fastText binary install at `fastTextInstall/bin/fasttext`. This path will used in the following setup scripts.

### 2. fastBPE

Compile the code with
```
git clone https://github.com/glample/fastBPE.git
cd fastBPE
g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
cd ..
```

This should create the fastBPE binary at `fastBPE/fast`. Similarly, this path will be used in the following setup scripts.

### 3. Moses

All you need to do is clone the repository (no installation needed)

```
git clone https://github.com/moses-smt/mosesdecoder.git
```

The moses base path `./mosesdecoder` will be used in following setup scripts.

### 4. pycrypto

Install with pip (no need to setup if you use Anaconda - it comes by default)
```
pip install pycrypto
```

## Downloading model files for pre-processing

Setting up the datasets requires downloading three fastText classifier model files

1. English language classifier - `lid.176.bin`
2. Yelp category classifier - `yelp_categories.model.bin`
3. Amazon category classifier - `amazon_categories.model.bin`

Download them into the empty `models` folder

```
cd models
wget https://dl.fbaipublicfiles.com/MultipleAttributeTextRewriting/lid.176.bin
wget https://dl.fbaipublicfiles.com/MultipleAttributeTextRewriting/yelp_categories.model.bin
wget https://dl.fbaipublicfiles.com/MultipleAttributeTextRewriting/amazon_categories.model.bin
cd ..
```

## Running the dataset setup scripts

### 1. FYelp

The data we are releasing is derived from the same base Yelp data as used in previous work, e.g. https://github.com/shrimai/Style-Transfer-Through-Back-Translation.

The `Yelp/yelp_pipeline.sh` file is the main run script to set up the dataset. At the topic of the bash script, you'll see pre-set paths to the fastBPE binary, fastText binary and the mosesdecoder folder based on the default setup instructions. Make sure they're all set correctly if you set things up differently.

Since the Yelp dataset [here](https://www.yelp.com/dataset/download) is not static, we cannot guarantee that the train/val/test splits that you obtain from downloading the dataset here, now or any time in the future, be the same as the one we used in our work, nor previous work such as https://github.com/shrimai/Style-Transfer-Through-Back-Translation. If you still want to proceed, download and place all of the raw JSON files in `dataset/yelp`

* business.json
* checkin.json
* review.json
* tip.json
* user.json

You can finally set things up by running

```
cd Yelp
mkdir -p ../dataset/yelp
bash yelp_pipeline.sh
```

Alternatively, if you already have a copy of the yelp reviews and attributes as two separate files, you can set things up by creating and copying them into the folder `dataset/yelp/processed/style_transfer`.

Our scripts expect that same format as in https://github.com/shrimai/Style-Transfer-Through-Back-Translation, namely: two files, one containing just the yelp reviews and the other containing its attributes. They must be named `fader.attr` and `fader.review` where `fader.attr` contains one review per line that already contains the tokenized and language filtered data using the same setup as in `Yelp/yelp_pipeline.sh`. `fader.attr` contains the corresponding attributes for each line in the review where columns must correspond to the follwing (1-indexed):

* Column 1. Gender (0 Male / 1 Female / 2 Unknown)
* Column 2. Sentiment (Number of stars)
* Column 3. Pos/Neg/Neutral Sentiment (-1 Neg / 0 Neutral / 1 Pos)
* Column 4. Restaurant (1 Restaurant / 0 Not a restaurant) 
* Column 5. Asian Food (1 Asian Food / 0 Not Asian food)
* Column 6. American Food (1 American Food / 0 Not American food)
* Column 7. Mexican Food (1 Mexican Food / 0 Not Mexican food)
* Column 8. Bar Food (1 Bar Food / 0 Not Bar food)

An example setup of this alternate pipeline would be

```
mkdir -p dataset/yelp/processed/style_transfer
cd dataset/yelp/processed/style_transfer
cp <path/to/reviews.txt> .
cp <path/to/attrs.txt> .
mv reviews.txt fader.review
mv attrs.txt fader.attr
cd ../../../../Yelp/
bash yelp_short_pipeline.sh
```

This should write the automatic train/val/test splits to

* dataset/yelp/processed/style_transfer/train.fader.with_cat.proc.<40000/60000/80000>
* dataset/yelp/processed/style_transfer/valid.fader.with_cat.proc.<40000/60000/80000>
* dataset/yelp/processed/style_transfer/test.with_cat.proc.<40000/60000/80000>

In each file, the first column is the review and the subsequent columns represent some of its attributes.

The important column attributes (1-indexed) are

* Column 1. Review text
* Column 2. Gender (0 Male / 1 Female / 2 Unknown)
* Column 3. Sentiment (Number of stars)
* Column 4. Pos/Neg/Neutral Sentiment (-1 Neg / 0 Neutral / 1 Pos)
* Column 5. Restaurant (1 Restaurant / 0 Not a restaurant) 
* Column 6. Asian Food (1 Asian Food / 0 Not Asian food)
* Column 7. American Food (1 American Food / 0 Not American food)
* Column 8. Mexican Food (1 Mexican Food / 0 Not Mexican food)
* Column 9. Bar Food (1 Bar Food / 0 Not Bar food)
* Column 10. Dessert Food (1 Dessert Food / 0 Not Dessert food)
* Column 34. Categorical Food Category (Automatically scored using a classifier)

It also creates a parallel human reference test in `human_test/yelp/proc`

### 2. Amazon

The `Amazon/amazon_pipeline.sh` file is the main run script to set up the dataset. At the top of the bash script, you'll see pre-set paths to the fastBPE binary, fastText binary and the mosesdecoder folder based on the default setup instructions. Make sure they're all set correctly if you set things up differently.

NOTE: This dataset is an order of magnitude larger than Yelp and will take pretty long to set up (~12 hours)

To run setup,

```
cd Amazon
mkdir -p ../dataset/amazon
bash amazon_pipeline.sh
```

This should write the automatic train/val/test splits to

* dataset/amazon/processed/style_transfer/train.fader.with_cat.proc.<40000/60000/80000>
* dataset/amazon/processed/style_transfer/valid.fader.with_cat.proc.<40000/60000/80000>
* dataset/amzon/processed/style_transfer/test.with_cat.proc.<40000/60000/80000>

In each file, the first column is the review and the subsequent columns represent some of its attributes.

The important column attributes (1-indexed) are

* Column 1. Review text
* Column 2. Gender (0 Male / 1 Female / 2 Unknown)
* Column 3. Sentiment (Number of stars)
* Column 4. Pos/Neg/Neutral Sentiment (-1 Neg / 0 Neutral / 1 Pos)
* Column 5. Books (1 Books / 0 Not Books)
* Column 6. Electronics (1 Electronics / 0 Not Electronics)
* Column 7. Movies (1 Movies / 0 Not Movies)
* Column 8. Clothing (1 Clothing / 0 Not Clothing)
* Column 9. Music (1 Music / 0 Not Music)
* Column 35. Categorical Review Category (Automatically scored using a classifier)

It also creates a parallel human reference test in `human_test/amazon/proc`
