# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import time
import argparse

from src.data.loader import check_all_data_params, load_data
from src.utils import bool_flag, attr_flag, initialize_exp
from src.model import check_ae_model_params, build_ae_model
from src.trainer import TrainerAE
from src.evaluator import EvaluatorAE


import signal
def ignore(signum, frame):
    print("Ignoring signal SIGTERM")
signal.signal(signal.SIGTERM, ignore)


# parse parameters
parser = argparse.ArgumentParser(description='Style transfer')
parser.add_argument("--exp_name", type=str, default="",
                    help="Experiment name")
parser.add_argument("--exp_id", type=str, default="",
                    help="Experiment ID")
parser.add_argument("--dump_path", type=str, default="",
                    help="Experiment dump path")
parser.add_argument("--save_periodic", type=str, default='',
                    help="Save the model periodically, and under certain conditions ('', '5', '5,self_bleu_test:+15,ftt_clf_test:+80')")
parser.add_argument("--seed", type=int, default=-1,
                    help="Random generator seed (-1 for random)")
# autoencoder parameters
parser.add_argument("--emb_dim", type=int, default=256,
                    help="Embedding layer size")
parser.add_argument("--n_enc_layers", type=int, default=2,
                    help="Number of layers in the encoders")
parser.add_argument("--n_dec_layers", type=int, default=2,
                    help="Number of layers in the decoders")
parser.add_argument("--hidden_dim", type=int, default=512,
                    help="Hidden layer size")
parser.add_argument("--lstm_proj", type=bool_flag, default=False,
                    help="Projection layer between decoder LSTM and output layer")
parser.add_argument("--dropout", type=float, default=0,
                    help="Dropout")
parser.add_argument("--label-smoothing", type=float, default=0,
                    help="Label smoothing")
parser.add_argument("--attention", type=bool_flag, default=True,
                    help="Use an attention mechanism")
parser.add_argument("--pool_latent", type=str, default='max,5',
                    help="Pool latent state representation - LSTM with attention only ('', 'max,3', 'avg,5')")
if not parser.parse_known_args()[0].attention:
    parser.add_argument("--enc_dim", type=int, default=512,
                        help="Latent space dimension")
    parser.add_argument("--proj_mode", type=str, default="last",
                        help="Projection mode (proj / pool / last)")
    parser.add_argument("--init_encoded", type=bool_flag, default=False,
                        help="Initialize the decoder with the encoded state. Append it to each input embedding otherwise.")
else:
    parser.add_argument("--transformer", type=bool_flag, default=False,
                        help="Use transformer architecture + attention mechanism")
    if parser.parse_known_args()[0].transformer:
        parser.add_argument("--transformer_ffn_emb_dim", type=int, default=512,
                            help="Transformer fully-connected hidden dim size")
        parser.add_argument("--attention_dropout", type=float, default=0,
                            help="attention_dropout")
        parser.add_argument("--relu_dropout", type=float, default=0,
                            help="relu_dropout")
        parser.add_argument("--encoder_attention_heads", type=int, default=8,
                            help="encoder_attention_heads")
        parser.add_argument("--decoder_attention_heads", type=int, default=8,
                            help="decoder_attention_heads")
        parser.add_argument("--encoder_normalize_before", type=bool_flag, default=False,
                            help="encoder_normalize_before")
        parser.add_argument("--decoder_normalize_before", type=bool_flag, default=False,
                            help="decoder_normalize_before")
    else:
        parser.add_argument("--input_feeding", type=bool_flag, default=False,
                            help="Input feeding")
parser.add_argument("--share_encdec_emb", type=bool_flag, default=False,
                    help="Share encoder embeddings / decoder embeddings")
parser.add_argument("--share_decpro_emb", type=bool_flag, default=False,
                    help="Share decoder embeddings / decoder output projection")
# attribute feeding
parser.add_argument("--bos_attr", type=str, default='avg',
                    help="Beginning of sentence attribute embedding")
parser.add_argument("--bias_attr", type=str, default='avg',
                    help="Attribute bias")
# encoder input perturbation
parser.add_argument("--word_shuffle", type=float, default=3,
                    help="Randomly shuffle input words (0 to disable)")
parser.add_argument("--word_dropout", type=float, default=0.1,
                    help="Randomly dropout input words (0 to disable)")
parser.add_argument("--word_blank", type=float, default=0.1,
                    help="Randomly blank input words (0 to disable)")
# discriminator parameters
parser.add_argument("--dis_layers", type=int, default=3,
                    help="Number of hidden layers in the discriminator")
parser.add_argument("--dis_hidden_dim", type=int, default=128,
                    help="Discriminator hidden layers dimension")
parser.add_argument("--dis_dropout", type=float, default=0,
                    help="Discriminator dropout")
parser.add_argument("--dis_clip", type=float, default=0,
                    help="Clip discriminator weights (0 to disable)")
parser.add_argument("--dis_smooth", type=float, default=0,
                    help="GAN smooth predictions")
parser.add_argument("--dis_input_proj", type=bool_flag, default=False,
                    help="Feed the discriminator with the projected output (attention only)")
parser.add_argument("--disc_lstm_dim", type=int, default=128,
                    help="Discriminator LSTM dimension")
parser.add_argument("--disc_lstm_layers", type=int, default=2,
                    help="Number of layers in the discriminator LSTM")
parser.add_argument("--disc_anneal_iterations", type=int, default=10,
                    help="Number of iterations to anneal the discriminator coefficient")
parser.add_argument("--adv_emb", type=bool_flag, default=False,
                    help="Adversarial training on embeddings")
# dataset
parser.add_argument("--attributes", type=attr_flag, default="binary_sentiment",
                    help="Attributes")
parser.add_argument("--mono_dataset", type=str, default="",
                    help="Monolingual dataset ('train_path,valid_path,test_path')")
parser.add_argument("--n_mono", type=int, default=-1,
                    help="Number of monolingual sentences (-1 for everything)")
parser.add_argument("--max_len", type=int, default=175,
                    help="Maximum length of sentences (after BPE)")
parser.add_argument("--max_vocab", type=int, default=-1,
                    help="Maximum vocabulary size (-1 to disable)")
# training steps
parser.add_argument("--n_dis", type=int, default=0,
                    help="Number of discriminator training iterations")
parser.add_argument("--train_ae", type=bool_flag, default=True,
                    help="Train denoising autoencoder")
parser.add_argument("--train_bt", type=bool_flag, default=True,
                    help="Train with online back-translation")
parser.add_argument("--otf_temperature", type=str, default="0.7",
                    help="Temperature for sampling back-translations (0 for greedy decoding)")
parser.add_argument("--otf_backprop_temperature", type=float, default=-1,
                    help="Back-propagate through the encoder (-1 to disable, temperature otherwise)")
parser.add_argument("--otf_sync_params_every", type=int, default=1000, metavar="N",
                    help="Number of updates between synchronizing params")
parser.add_argument("--otf_num_processes", type=int, default=16, metavar="N",
                    help="Number of processes to use for OTF generation")
parser.add_argument("--otf_update_enc", type=bool_flag, default=True,
                    help="Update the encoder during back-translation training")
parser.add_argument("--otf_update_dec", type=bool_flag, default=True,
                    help="Update the decoder during back-translation training")
# language model training
parser.add_argument("--lm_before", type=int, default=0,
                    help="Training steps with language model pretraining (0 to disable)")
parser.add_argument("--lm_after", type=int, default=0,
                    help="Keep training the language model during MT training (0 to disable)")
# training parameters
parser.add_argument("--balanced_train", type=bool_flag, default=False,
                    help="Sample sentences in a balanced way during training.")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size")
parser.add_argument("--group_by_size", type=bool_flag, default=False,
                    help="Sort sentences by size during the training")
parser.add_argument("--lambda_ae", type=str, default="1.0",
                    help="Cross-entropy reconstruction coefficient (autoencoding)")
parser.add_argument("--lambda_bt", type=str, default="1.0",
                    help="Cross-entropy reconstruction coefficient (on-the-fly back-translation parallel data)")
parser.add_argument("--lambda_dis", type=str, default="0",
                    help="Discriminator loss coefficient")
parser.add_argument("--lambda_lm", type=str, default="0",
                    help="Language model loss coefficient")
parser.add_argument("--enc_optimizer", type=str, default="adam,lr=0.0003",
                    help="Encoder optimizer (SGD / RMSprop / Adam, etc.)")
parser.add_argument("--dec_optimizer", type=str, default="enc_optimizer",
                    help="Decoder optimizer (SGD / RMSprop / Adam, etc.)")
parser.add_argument("--dis_optimizer", type=str, default="rmsprop,lr=0.0005",
                    help="Discriminator optimizer (SGD / RMSprop / Adam, etc.)")
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
parser.add_argument("--pretrained_emb", type=str, default="",
                    help="Reload pretrained source and target word embeddings")
parser.add_argument("--pretrained_out", type=bool_flag, default=False,
                    help="Pretrain the decoder output projection matrix")
parser.add_argument("--reload_model", type=str, default="",
                    help="Reload a pretrained model")
parser.add_argument("--reload_enc", type=bool_flag, default=False,
                    help="Reload a pretrained encoder")
parser.add_argument("--reload_dec", type=bool_flag, default=False,
                    help="Reload a pretrained decoder")
parser.add_argument("--reload_dis", type=bool_flag, default=False,
                    help="Reload a pretrained discriminator")
# freeze network parameters
parser.add_argument("--freeze_enc_emb", type=bool_flag, default=False,
                    help="Freeze encoder embeddings")
parser.add_argument("--freeze_dec_emb", type=bool_flag, default=False,
                    help="Freeze decoder embeddings")
# evaluation
parser.add_argument("--n_eval_sentences", type=int, default=-1,
                    help="Number of experiments to consider for evaluation (-1 for everything)")
parser.add_argument("--eval_ftt_clf", type=str, default="",
                    help="Reload pretrained fastText classifiers for evaluation")
parser.add_argument("--eval_cnn_clf", type=str, default="",
                    help="Reload a pretrained CNN classifier for evaluation")
parser.add_argument("--eval_lm", type=str, default="",
                    help="Reload a pretrained language model for evaluation")
parser.add_argument("--eval_para", type=str, default="",
                    help="Evaluate on parallel sentences (name1:src_path:ref0_path,...,ref3_path;name2:src_path:ref0_path,...,ref3_path)")
parser.add_argument("--eval_only", type=bool_flag, default=False,
                    help="Only run evaluations")
parser.add_argument("--beam_size", type=int, default=0,
                    help="Beam width (<= 0 means greedy)")
parser.add_argument("--length_penalty", type=float, default=1.0,
                    help="Length penalty: <1.0 favors shorter, >1.0 favors longer sentences")
parser.add_argument("--bleu_script_path", type=str, required=True,
                    help="Path to moses bleu script.")
params = parser.parse_args()


if __name__ == '__main__':

    # check parameters
    params.name = params.exp_name
    assert len(params.name.strip()) > 0
    check_all_data_params(params)
    check_ae_model_params(params)

    # initialize / load data / build model
    logger = initialize_exp(params)
    data = load_data(params)
    encoder, decoder, discriminator, lm = build_ae_model(params, data)

    # trainer / checkpoint / evaluator
    trainer = TrainerAE(encoder, decoder, discriminator, lm, data, params)
    trainer.reload_checkpoint()
    trainer.test_sharing()  # check parameters sharing
    evaluator = EvaluatorAE(trainer, data, params, params.bleu_script_path)

    # only evaluate the model
    if params.eval_only:
        evaluator.run_all_evals(0)
        exit()

    # language model pretraining
    if params.lm_before > 0:
        assert False  # TODO: implement
        logger.info("Pretraining language model for %i iterations ..." % params.lm_before)
        trainer.n_sentences = 0
        for _ in range(params.lm_before):
            trainer.lm_step()
            trainer.iter()

    # start training
    for _ in range(trainer.epoch, params.max_epoch):

        logger.info("=================== Starting epoch %i ... ===================" % trainer.epoch)

        trainer.n_sentences = 0

        while trainer.n_sentences < params.epoch_size:

            # # discriminator training
            # for _ in range(params.n_dis):
            #     trainer.discriminator_step()

            # # language model training
            # if params.lambda_lm > 0:
            #     for _ in range(params.lm_after):
            #         trainer.lm_step()

            # autoencoder training (monolingual data)
            if params.lambda_ae > 0:
                trainer.enc_dec_step(params.lambda_ae, attr_label=trainer.get_attr_label())

            # autoencoder training (on the fly back-translation)
            if params.lambda_bt > 0:

                # initialize CPU threads
                if not hasattr(params, 'otf_batch_gen_started'):
                    otf_iterator = trainer.otf_bt_gen_async()
                    params.otf_batch_gen_started = True

                # update CPU threads sampling generation temperature
                trainer.otf_sync_temperature()

                # update model parameters on subprocesses
                if trainer.n_iter % params.otf_sync_params_every == 0:
                    trainer.otf_sync_params()

                # get training batch from CPU
                before_gen = time.time()
                batches = next(otf_iterator)
                trainer.gen_time += time.time() - before_gen

                # training
                for batch in batches:
                    trainer.otf_bt(batch, params.lambda_bt, params.otf_backprop_temperature)

            trainer.iter()

        # end epoch
        logger.info("====================== End of epoch %i ======================" % trainer.epoch)

        # evaluate discriminator / perplexity / BLEU
        scores = evaluator.run_all_evals(trainer.epoch)

        # end epoch
        trainer.end_epoch(scores)
        trainer.test_sharing()
