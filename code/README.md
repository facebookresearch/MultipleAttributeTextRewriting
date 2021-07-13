# Code for Multiple-Attribute Text Rewriting

## 1. Dataset creation

Please follow instructions in data/README.md to create training, validation and test datasets for Yelp and Amazon. In the rest of this README, we will use Yelp as an example for how to run things.

After dataset creation, you should end up with a folder `../data/dataset/yelp/processed/style_transfer/` that contains processed datasets for Yelp.

## 2. Train Sentiment & Category Classifiers for Evaluation while Training

```bash
mkdir ../data/dataset/yelp/processed/style_transfer/classifier_training
cd ../data/dataset/yelp/processed/style_transfer/
```

### Create data for fastText training of Sentiment, Binary Sentiment & Categories

```bash
awk -F "\t" '{print $1" __label__"$3}' train.fader.with_cat.proc.40000 > classifier_training/train.sentiment.40000.txt
awk -F "\t" '{print $1" __label__"$4}' train.fader.with_cat.proc.40000 > classifier_training/train.binary_sentiment.40000.txt
sed -i '/__label__0/d' classifier_training/train.binary_sentiment.40000.txt
awk -F "\t" '{print $1" __label__"$34}' train.fader.with_cat.proc.40000 > classifier_training/train.categories.40000.txt

awk -F "\t" '{print $1" __label__"$3}' valid.fader.with_cat.proc.40000 > valid sentiment.40000.txt
awk -F "\t" '{print $1" __label__"$4}' valid.fader.with_cat.proc.40000 > valid.binary_sentiment.40000.txt
sed -i '/__label__0/d' valid.binary_sentiment.40000.txt
awk -F "\t" '{print $1" __label__"$34}' valid.fader.with_cat.proc.40000 > valid.categories.40000.txt
```

### Train and Evaluate fastText classifiers

```bash
fasttext supervised -wordNgrams 4 -minn 3 -maxn 3 -input classifier_training/train.binary_sentiment.40000.txt -output classifier_training/fasttext.binary_sentiment.40000

fasttext test classifier_training/fasttext.binary_sentiment.40000.bin classifier_training/valid.binary_sentiment.40000.txt
```

You should expect accuracies around ~97% for binary sentiment classification
```
N	8948
P@1	0.978
R@1	0.978
```

```bash
fasttext supervised -wordNgrams 4 -minn 3 -maxn 3 -input classifier_training/train.categories.40000.txt -output classifier_training/fasttext.categories.40000

fasttext test classifier_training/fasttext.categories.40000.bin classifier_training/valid.categories.40000.txt
```

You should expect accuracies around ~85% for binary sentiment classification

```
N       10000
P@1     0.852
R@1     0.852
```

## 3. Binarize Proessed Data

Once you have the processed data, we will binarze the data into `.pth` files to load data quickly.

```bash
bash code/binarize_yelp_data.sh
```

This will generate files that we will pass to the training script.

```bash
data/dataset/yelp/processed/style_transfer/train.fader.with_cat.proc.<bpe_codes>.pth
data/dataset/yelp/processed/style_transfer/valid.fader.with_cat.proc.<bpe_codes>.pth
data/dataset/yelp/processed/style_transfer/test.fader.with_cat.proc.<bpe_codes>.pth
```

## 4. Training models

To train models, use the `code/main-parallel.py` script.

### Binary sentiment

```bash
mkdir models/style_transfer
python main-parallel.py --exp_name test \
    --dump_path models/style_transfer \
    --mono_dataset ../data/dataset/yelp/processed/style_transfer/train.fader.with_cat.proc.40000.pth,../data/dataset/yelp/processed/style_transfer/valid.fader.with_cat.proc.40000.pth,../data/dataset/yelp/processed/style_transfer/test.fader.with_cat.proc.40000.pth \
    --attributes binary_sentiment \
    --n_mono -1 \
    --lambda_ae 1.0 \
    --lambda_bt 0.6 \
    --train_ae true \
    --train_bt true \
    --eval_ftt_clf binary_sentiment:../data/dataset/yelp/processed/style_transfer/fasttext.binary_sentiment.40000.bin \
    --bleu_script_path ../data/mosesdecoder/scripts/generic/multi-bleu.perl \
    --balanced_train true
```

This will create a folder inside `models/style_transfer` that will contain the training log files, model checkpoints and generations on the valid/test sets.

The log files will contain validation and test self-BLEU as well as classifier accuracies for example:

```bash
INFO - 06/30/21 05:45:48 - 5:46:04 - ====================== End of epoch 8 ======================
INFO - 06/30/21 05:45:48 - 5:46:04 - Evaluating sentences using pretrained classifiers (valid) ...
INFO - 06/30/21 05:46:43 - 5:46:59 - BLEU - valid - binary_sentiment -            ->         -1 |          1 |      Total
INFO - 06/30/21 05:46:43 - 5:46:59 - BLEU - valid - binary_sentiment -         -1 ->      35.94 |      31.19 |      33.56
INFO - 06/30/21 05:46:47 - 5:47:03 - BLEU - valid - binary_sentiment -          1 ->      34.81 |      39.23 |      37.02
INFO - 06/30/21 05:46:47 - 5:47:03 - BLEU - valid - binary_sentiment: 35.292
INFO - 06/30/21 05:46:47 - 5:47:03 - BLEU - valid: 35.292
INFO - 06/30/21 05:46:47 - 5:47:03 - Accu - valid - binary_sentiment -            ->         -1 |          1 |      Total
INFO - 06/30/21 05:46:47 - 5:47:03 - Accu - valid - binary_sentiment -         -1 ->      98.13 |      50.76 |      74.44
INFO - 06/30/21 05:46:47 - 5:47:03 - Accu - valid - binary_sentiment -          1 ->      85.79 |      99.89 |      92.84
INFO - 06/30/21 05:46:47 - 5:47:03 - Accu - valid - binary_sentiment: 83.642
INFO - 06/30/21 05:46:47 - 5:47:03 - Confusion matrix for binary_sentiment:
INFO - 06/30/21 05:46:47 - 5:47:03 - [[[ 891   17]
                                       [ 129  779]]

                                      [[1769 1716]
                                       [   4 3481]]]
INFO - 06/30/21 05:46:47 - 5:47:03 - Accu - valid: 83.642
```

NOTE: In our experiments, models significantly benefit from training for long periods of time (~5-6 days on a single GPU) and are fairly sensitive to the choice of hyperparameters that control the accuracy vs BLEU trade-off. The logs above are just an example of what you should expect after ~6 hours of training.

### Restaurant Cuisine Type

```bash
mkdir models/style_transfer
python main-parallel.py --exp_name test \
    --dump_path models/style_transfer \
    --mono_dataset ../data/dataset/yelp/processed/style_transfer/train.fader.with_cat.proc.40000.pth,../data/dataset/yelp/processed/style_transfer/valid.fader.with_cat.proc.40000.pth,../data/dataset/yelp/processed/style_transfer/test.fader.with_cat.proc.40000.pth \
    --attributes type \
    --n_mono -1 \
    --lambda_ae 1.0 \
    --lambda_bt 0.6 \
    --train_ae true \
    --train_bt true \
    --eval_ftt_clf type:../data/dataset/yelp/processed/style_transfer/fasttext.categories.40000.bin \
    --bleu_script_path ../data/mosesdecoder/scripts/generic/multi-bleu.perl \
    --balanced_train true
```

### Restaurant Cuisine Type and Sentiment Together

```bash
mkdir models/style_transfer
python main-parallel.py --exp_name test \
    --dump_path models/style_transfer \
    --mono_dataset ../data/dataset/yelp/processed/style_transfer/train.fader.with_cat.proc.40000.pth,../data/dataset/yelp/processed/style_transfer/valid.fader.with_cat.proc.40000.pth,../data/dataset/yelp/processed/style_transfer/test.fader.with_cat.proc.40000.pth \
    --attributes binary_sentiment,type \
    --n_mono -1 \
    --lambda_ae 1.0 \
    --lambda_bt 0.6 \
    --train_ae true \
    --train_bt true \
    --eval_ftt_clf binary_sentiment:../data/dataset/yelp/processed/style_transfer/fasttext.binary_sentiment.40000.bin,type:../data/dataset/yelp/processed/style_transfer/fasttext.category.40000.bin \
    --bleu_script_path ../data/mosesdecoder/scripts/generic/multi-bleu.perl
```
