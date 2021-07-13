# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pdb
import argparse
import numpy as np

from src.data.loader import check_mono_data_params, load_data
from src.utils import bool_flag, attr_flag, initialize_exp
from src.model.classifier import check_classifier_params, build_classifier_model
from src.trainer import TrainerClassifier
from src.evaluator import EvaluatorClassifier


# parse parameters
parser = argparse.ArgumentParser(description='Attribute-based classifier - Training')
parser.add_argument("--exp_name", type=str, default="",
                    help="Experiment name")
parser.add_argument("--exp_id", type=str, default="",
                    help="Experiment ID")
parser.add_argument("--dump_path", type=str, default="",
                    help="Experiment dump path")
parser.add_argument("--save_periodic", type=str, default='',
                    help="Save the model periodically, and under certain conditions ('', '5', '5,self_bleu_test+15,ftt_clf_test+80')")
parser.add_argument("--seed", type=int, default=-1,
                    help="Random generator seed (-1 for random)")
# classifier parameters
parser.add_argument("--emb_dim", type=int, default=256,
                    help="Embedding layer size")
parser.add_argument("--clf_n_layers", type=int, default=1,
                    help="Number of layers")
parser.add_argument("--clf_n_kernels", type=int, default=32,
                    help="Number of kernels")
parser.add_argument("--clf_kernel_size", type=int, default=5,
                    help="Number of kernels")
parser.add_argument("--clf_dropout", type=float, default=0,
                    help="Dropout")
# dataset
parser.add_argument("--attributes", type=attr_flag, default="gender,lifestage,age_bracket",
                    help="Attributes")
parser.add_argument("--mono_dataset", type=str, default="",
                    help="Monolingual dataset ('train_path,valid_path,test_path')")
parser.add_argument("--n_mono", type=int, default=0,
                    help="Number of monolingual sentences (-1 for everything)")
parser.add_argument("--max_len", type=int, default=175,
                    help="Maximum length of sentences (after BPE)")
parser.add_argument("--max_vocab", type=int, default=-1,
                    help="Maximum vocabulary size (-1 to disable)")
# training parameters
parser.add_argument("--balanced_train", type=bool_flag, default=False,
                    help="Sample sentences in a balanced way during training.")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size")
parser.add_argument("--group_by_size", type=bool_flag, default=True,
                    help="Sort sentences by size during the training")
parser.add_argument("--clf_optimizer", type=str, default="adam,lr=0.0003",
                    help="Encoder optimizer (SGD / RMSprop / Adam, etc.)")
parser.add_argument("--clip_grad_norm", type=float, default=5,
                    help="Clip gradients norm (0 to disable)")
parser.add_argument("--epoch_size", type=int, default=100000,
                    help="Epoch size / evaluation frequency")
parser.add_argument("--max_epoch", type=int, default=100000,
                    help="Maximum epoch size")
parser.add_argument("--stop_crit", type=str, default="",
                    help="Stopping criterion, and number of non-increase before stopping the experiment")
parser.add_argument("--metrics", type=str, default="",
                    help="Validation metrics")
# reload models
parser.add_argument("--reload_model", type=str, default="",
                    help="Reload a pretrained model")
# evaluation
parser.add_argument("--n_eval_sentences", type=int, default=-1,
                    help="Number of experiments to consider for evaluation (-1 for everything)")
parser.add_argument("--eval_only", type=int, default=0,
                    help="Only run evaluations")
params = parser.parse_args()


if __name__ == '__main__':

    # check parameters
    params.name = params.exp_name
    assert len(params.name.strip()) > 0
    check_mono_data_params(params)
    check_classifier_params(params)

    # initialize / load data / build model
    logger = initialize_exp(params)
    data = load_data(params, mono_only=True)
    classifier = build_classifier_model(params, data['dico'])

    # trainer / checkpoint / evaluator
    trainer = TrainerClassifier(classifier, data, params)
    trainer.reload_checkpoint()
    evaluator = EvaluatorClassifier(trainer, data, params)

    # only run evaluations
    if params.eval_only:
        if params.eval_only == 1:
            evaluator.run_all_evals(0)
        if params.eval_only == 2:
            pdb.set_trace()
        exit()

    # start training
    for _ in range(trainer.epoch, params.max_epoch):

        logger.info("=================== Starting epoch %i ... ===================" % trainer.epoch)

        trainer.n_sentences = 0

        while trainer.n_sentences < params.epoch_size:
            trainer.clf_step(attr_label=trainer.get_attr_label())
            trainer.iter()

        logger.info("====================== End of epoch %i ======================" % trainer.epoch)

        # evaluate classifier
        scores = evaluator.run_all_evals(trainer.epoch)

        # end epoch
        trainer.end_epoch(scores)
