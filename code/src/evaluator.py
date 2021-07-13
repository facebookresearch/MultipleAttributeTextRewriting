# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import json
import subprocess
from collections import OrderedDict
from logging import getLogger
import numpy as np
import torch
from torch.nn import functional as F

from .utils import restore_segmentation


logger = getLogger()


class EvaluatorBase(object):

    def get_iterator(self, data_type, attr_label=None):
        """
        Create a new iterator for a dataset.
        """
        assert data_type in ['valid', 'test']
        return self.data['mono'][data_type].get_iterator(
            shuffle=False, group_by_size=True,
            n_sentences=self.n_eval_sentences, attr_label=attr_label
        )()

    def run_all_evals_(self, epoch):
        raise NotImplementedError

    def run_all_evals(self, epoch):
        """
        Run and log evaluations.
        """
        scores = self.run_all_evals_(epoch)
        # for k, v in scores.items():
        #     logger.info('%s -> %.6f' % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))
        return scores


class EvaluatorAE(EvaluatorBase):

    def __init__(self, trainer, data, params, bleu_script_path):
        """
        Initialize evaluator.
        """
        self.encoder = trainer.encoder
        self.decoder = trainer.decoder
        self.data = data
        self.dico = data['dico']
        self.params = params
        self.bleu_script_path = bleu_script_path

        # reload a pretrained CNN classifier
        if params.eval_cnn_clf != '':
            assert os.path.isfile(params.eval_cnn_clf)
            logger.info("Reloading pretrained CNN classifier from %s ..." % params.eval_cnn_clf)
            self.cnn_clf = torch.load(params.eval_cnn_clf)['clf']
            assert self.cnn_clf.attributes == self.params.attributes
            assert self.cnn_clf.dico == self.dico
        else:
            self.cnn_clf = None

        # reload a pretrained fastText classifier
        if params.eval_ftt_clf != '':
            import fasttext
            assert set(params.eval_ftt_clf.keys()) == set(params.attributes)
            self.ftt_clfs = {
                attr: fasttext.load_model(path) if not attr.startswith('length_') else ''
                for attr, path in params.eval_ftt_clf.items()
            }
        else:
            self.ftt_clfs = None

        # reload a pretrained language model
        if params.eval_lm != '':
            assert os.path.isfile(params.eval_lm)
            logger.info("Reloading pretrained language model from %s ..." % params.eval_lm)
            self.lm = torch.load(params.eval_lm)['lm']
            assert self.lm.attributes == self.params.attributes
            assert self.lm.dico == self.dico
        else:
            self.lm = None

        # number of sentences for evaluation
        self.n_eval_sentences = params.n_eval_sentences
        assert self.n_eval_sentences == -1 or self.n_eval_sentences >= 1

        # create directory to store hypothesis
        params.hyp_path = os.path.join(params.dump_path, 'hypothesis')
        subprocess.Popen('mkdir -p %s' % params.hyp_path, shell=True).wait()

        # export source sentences
        self.export_src_sentences('valid')
        self.export_src_sentences('test')

        # export evaluation source / references.
        self.export_eval_sentences()
    
    def eval_moses_bleu(self, ref, hyp):
        """
        Given a file of hypothesis and reference files,
        evaluate the BLEU score using Moses scripts.
        """
        assert os.path.isfile(ref) or os.path.isfile(ref + '0')
        assert os.path.isfile(hyp)
        command = self.bleu_script_path + ' %s < %s'
        p = subprocess.Popen(command % (ref, hyp), stdout=subprocess.PIPE, shell=True)
        result = p.communicate()[0].decode("utf-8")
        if result.startswith('BLEU'):
            return float(result[7:result.index(',')])
        else:
            logger.warning('Impossible to parse BLEU score! "%s"' % result)
            return -1


    def export_src_sentences(self, data_type):
        """
        Export source sentences.
        """
        params = self.params

        # for each attribute
        for attr_id, attr in enumerate(params.attributes):

            # for each label
            for label_id, label in enumerate(params.attr_values[attr]):

                # save sentences
                txt = []
                for (sent, lengths, _) in self.get_iterator(data_type, (attr_id, label_id)):
                    txt.extend(convert_to_text(sent, lengths, self.dico, params))

                # export sentences
                filename = 'ref.%s.%s.%s' % (data_type, attr, label)
                ref_path = os.path.join(params.hyp_path, filename)
                with open(ref_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(txt) + '\n')

                # restore BPE segmentation
                restore_segmentation(ref_path)

    def export_eval_sentences(self):
        """
        Export evaluation source / references.
        """
        params = self.params

        assert not (params.eval_para == '') ^ ('eval' not in self.data)
        if 'eval' not in self.data:
            return

        for name, (src_data, ref_data) in self.data['eval'].items():

            assert all([len(x) == len(src_data) for x in ref_data])

            for i, data in enumerate([src_data] + ref_data):

                # save sentences
                txt = []
                for (sent, lengths, _) in data.get_iterator(shuffle=False, group_by_size=False)():
                    txt.extend(convert_to_text(sent, lengths, self.dico, params))

                # export sentences
                filename = 'eval.%s.%s' % (name, 'src' if i == 0 else 'ref%i' % (i - 1))
                path = os.path.join(params.hyp_path, filename)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(txt) + '\n')

                # restore BPE segmentation
                restore_segmentation(path)

    def eval_swap_bleu_clf(self, data_type, scores):
        """
        Classify sentences with swapped attributes using pretrained classifiers.
        """
        logger.info("Evaluating sentences using pretrained classifiers (%s) ..." % data_type)
        assert data_type in ['valid', 'test']
        self.encoder.eval()
        self.decoder.eval()
        if self.cnn_clf is not None:
            self.cnn_clf.eval()
        if self.lm is not None:
            self.lm.eval()
        params = self.params

        # initialize confusion matrices
        confusion_cnn = []
        confusion_ftt = []
        for a in params.attributes:
            n_attr = len(params.attr_values[a])
            confusion_cnn.append(np.zeros((n_attr, n_attr, n_attr), dtype=np.int32))
            confusion_ftt.append(np.zeros((n_attr, n_attr, n_attr), dtype=np.int32))

        # initialize hypothesis sentences
        hypothesis = {
            a: {(l1, l2): [] for l1 in params.attr_values[a] for l2 in params.attr_values[a]}
            for a in params.attributes
        }

        offset = 0
        # for each attribute
        for attr_id, attr in enumerate(params.attributes):

            # number of labels for this attribute
            n_attr = len(params.attr_values[attr])

            # for each label
            for label_id, label in enumerate(params.attr_values[attr]):

                # for all sentences with this label
                for (sent1, len1, attr1) in self.get_iterator(data_type, (attr_id, label_id)):

                    # check attribute / cuda batch / encode sentence
                    assert (attr1[:, attr_id] - offset == label_id).sum() == attr1.size(0)
                    sent1, attr1 = sent1.cuda(), attr1.cuda()
                    encoded = self.encoder(sent1, len1)

                    # try all labels
                    for new_label_id, new_label in enumerate(params.attr_values[attr]):

                        # update attribute / generate hypothesis with new attributes
                        attr1[:, attr_id] = new_label_id + offset
                        max_len = int(1.5 * len1.max() + 10)
                        sent2, len2, _ = self.decoder.generate(encoded, attr1, max_len=max_len)

                        # save hypothesis
                        hypothesis[attr][(label, new_label)].append((sent2, len2, attr1.clone()))

                        # CNN classifier
                        if self.cnn_clf is not None:
                            clf_scores = self.cnn_clf(sent2, len2)
                            predictions = clf_scores[:, offset:offset + n_attr].cpu().numpy().argmax(1)
                            for p in predictions:
                                confusion_cnn[attr_id][label_id, new_label_id, p] += 1

                        # fastText classifier
                        if self.ftt_clfs is not None:
                            # length label (small hack to include length in a fastText classifier)
                            if attr.startswith('length_'):
                                predictions = (len2 - 2).float().div(params.bucket_size).sub(1).clamp(0, n_attr - 1).long()
                            else:
                                samples = convert_to_text(sent2, len2, self.dico, params)
                                # get top 5 predictions
                                predictions = self.ftt_clfs[attr].predict(samples, k=5)[0]
                                ##
                                # this section is to deal with -1 / 1 labels for binary sentiment classifier. TODO: remove in the end
                                if attr == 'binary_sentiment':
                                    predictions = [[l.replace('__0', '__-1') for l in p] for p in predictions]
                                ##
                                # remove __label__ prefix and ignored labels (9 is the length of __label__)
                                predictions = [[l[9:] for l in p if l[9:] in params.attr_values[attr]][0] for p in predictions]
                                predictions = [params.attr_values[attr].index(p) for p in predictions]
                            for p in predictions:
                                confusion_ftt[attr_id][label_id, new_label_id, p] += 1

            offset += n_attr

        #
        # export references / hypothesis - compute self BLEU
        #
        PATTERN1 = 'BLEU - {:>5}: {:.3f}'
        PATTERN2 = 'BLEU - {:>5} - {:>10}: {:.3f}'
        PATTERN3 = 'BLEU - {:>5} - {:>10} - {:>10} -> {}'

        # for each attribute
        for attr in params.attributes:
            labels = params.attr_values[attr]

            # for each label
            for label_id, label in enumerate(labels):

                # for each new label
                for new_label_id, new_label in enumerate(labels):

                    # convert hypothesis to text
                    txt = []
                    for sent, lengths, _ in hypothesis[attr][(label, new_label)]:
                        txt.extend(convert_to_text(sent, lengths, self.dico, params))

                    # export hypothesis / restore BPE segmentation
                    filename = 'hyp.%s.%s.%s.%s.%i' % (data_type, attr, label, new_label, scores['epoch'])
                    hyp_path = os.path.join(params.hyp_path, filename)
                    with open(hyp_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(txt) + '\n')
                    restore_segmentation(hyp_path)

                    # new label self BLEU
                    filename = 'ref.%s.%s.%s' % (data_type, attr, label)
                    ref_path = os.path.join(params.hyp_path, filename)
                    bleu = self.eval_moses_bleu(ref_path, hyp_path)
                    scores['self_bleu_%s_%s_%s_%s' % (data_type, attr, label, new_label)] = bleu

                # label self BLEU
                bleus = [scores['self_bleu_%s_%s_%s_%s' % (data_type, attr, label, new_label)] for new_label in labels]
                bleu = np.mean(bleus)
                scores['self_bleu_%s_%s_%s' % (data_type, attr, label)] = bleu
                if label_id == 0:
                    logger.info(PATTERN3.format(data_type, attr, '', " | ".join(["%10s" % l for l in labels + ['Total']])))
                logger.info(PATTERN3.format(data_type, attr, label, " | ".join(["%10.2f" % b for b in bleus] + ["%10.2f" % bleu])))

            # attribute self BLEU
            bleu = np.mean([scores['self_bleu_%s_%s_%s' % (data_type, attr, label)] for label in labels])
            scores['self_bleu_%s_%s' % (data_type, attr)] = bleu
            logger.info(PATTERN2.format(data_type, attr, bleu))

        # overall self BLEU
        bleu = np.mean([scores['self_bleu_%s_%s' % (data_type, attr)] for attr in params.attributes])
        scores['self_bleu_%s' % data_type] = bleu
        logger.info(PATTERN1.format(data_type, bleu))

        #
        # evaluate language model perplexity
        #
        if self.lm is not None:

            PATTERN1 = 'PPL  - {:>5}: {:.3f}'
            PATTERN2 = 'PPL  - {:>5} - {:>10}: {:.3f}'
            PATTERN3 = 'PPL  - {:>5} - {:>10} - {:>10} -> {}'

            # for each attribute
            for attr in params.attributes:
                labels = params.attr_values[attr]

                # for each label
                for label_id, label in enumerate(labels):

                    # for each new label
                    for new_label_id, new_label in enumerate(labels):

                        total_loss = 0
                        total_words = 0

                        for sent, lengths, attributes in hypothesis[attr][(label, new_label)]:
                            log_probs = self.lm(sent[:-1], lengths - 1, attributes)
                            total_loss += F.cross_entropy(
                                log_probs.view(-1, self.params.n_words),
                                sent[1:].view(-1),
                                size_average=False
                            )
                            total_words += (lengths - 1).sum()

                        # new label perplexity
                        ppl = np.exp(total_loss.item() / total_words.item())
                        scores['ppl_%s_%s_%s_%s' % (data_type, attr, label, new_label)] = ppl

                    # label perplexity
                    ppls = [scores['ppl_%s_%s_%s_%s' % (data_type, attr, label, new_label)] for new_label in labels]
                    ppl = np.mean(ppls)
                    scores['ppl_%s_%s_%s' % (data_type, attr, label)] = ppl
                    if label_id == 0:
                        logger.info(PATTERN3.format(data_type, attr, '', " | ".join(["%10s" % l for l in labels + ['Total']])))
                    logger.info(PATTERN3.format(data_type, attr, label, " | ".join(["%10.2f" % b for b in ppls] + ["%10.2f" % ppl])))

                # attribute perplexity
                ppl = np.mean([scores['ppl_%s_%s_%s' % (data_type, attr, label)] for label in labels])
                scores['ppl_%s_%s' % (data_type, attr)] = ppl
                logger.info(PATTERN2.format(data_type, attr, ppl))

            # overall perplexity
            ppl = np.mean([scores['ppl_%s_%s' % (data_type, attr)] for attr in params.attributes])
            scores['ppl_%s' % data_type] = ppl
            logger.info(PATTERN1.format(data_type, ppl))

        #
        # report CNN classifier accuracy for each attribute
        #
        if self.cnn_clf is not None:

            PATTERN1 = 'Accu - {:>5}: {:.3f}'
            PATTERN2 = 'Accu - {:>5} - {:>10}: {:.3f}'
            PATTERN3 = 'Accu - {:>5} - {:>10} - {:>10} -> {}'

            # for each attribute
            for attr_id, attr in enumerate(params.attributes):
                labels = params.attr_values[attr]

                # for each new label
                for new_label_id, new_label in enumerate(labels):

                    # for each original label
                    for label_id, label in enumerate(labels):
                        correct = confusion_cnn[attr_id][label_id, new_label_id, new_label_id]
                        total = confusion_cnn[attr_id][label_id, new_label_id].sum()
                        accuracy = 100 * float(correct) / float(total)
                        scores['cnn_clf_%s_%s_%s_%s' % (data_type, attr, label, new_label)] = accuracy

                    # new label accuracy
                    accus = [scores['cnn_clf_%s_%s_%s_%s' % (data_type, attr, label, new_label)] for label in labels]
                    accu = np.mean(accus)
                    scores['cnn_clf_%s_%s_%s' % (data_type, attr, new_label)] = accu
                    if new_label_id == 0:
                        logger.info(PATTERN3.format(data_type, attr, '', " | ".join(["%10s" % l for l in labels + ['Total']])))
                    logger.info(PATTERN3.format(data_type, attr, new_label, " | ".join(["%10.2f" % a for a in accus] + ["%10.2f" % accu])))

                # attribute accuracy
                accu = np.mean([scores['cnn_clf_%s_%s_%s' % (data_type, attr, new_label)] for new_label in labels])
                scores['cnn_clf_%s_%s' % (data_type, attr)] = accu
                logger.info(PATTERN2.format(data_type, attr, accu))

                # log attribute confusion matrix
                logger.info("Confusion matrix for %s:" % attr)
                logger.info(confusion_cnn[attr_id])

            # overall accuracy
            accuracy = np.mean([scores['cnn_clf_%s_%s' % (data_type, a)] for a in params.attributes])
            scores['cnn_clf_%s' % data_type] = accuracy
            logger.info(PATTERN1.format(data_type, accuracy))

        if self.ftt_clfs is not None:

            PATTERN1 = 'Accu - {:>5}: {:.3f}'
            PATTERN2 = 'Accu - {:>5} - {:>10}: {:.3f}'
            PATTERN3 = 'Accu - {:>5} - {:>10} - {:>10} -> {}'

            # for each attribute
            for attr_id, attr in enumerate(params.attributes):
                labels = params.attr_values[attr]

                # for each new label
                for new_label_id, new_label in enumerate(labels):

                    # for each original label
                    for label_id, label in enumerate(labels):
                        correct = confusion_ftt[attr_id][label_id, new_label_id, new_label_id]
                        total = confusion_ftt[attr_id][label_id, new_label_id].sum()
                        accuracy = 100 * float(correct) / float(total)
                        scores['ftt_clf_%s_%s_%s_%s' % (data_type, attr, label, new_label)] = accuracy

                    # new label accuracy
                    accus = [scores['ftt_clf_%s_%s_%s_%s' % (data_type, attr, label, new_label)] for label in labels]
                    accu = np.mean(accus)
                    scores['ftt_clf_%s_%s_%s' % (data_type, attr, new_label)] = accu
                    if new_label_id == 0:
                        logger.info(PATTERN3.format(data_type, attr, '', " | ".join(["%10s" % l for l in labels + ['Total']])))
                    logger.info(PATTERN3.format(data_type, attr, new_label, " | ".join(["%10.2f" % a for a in accus] + ["%10.2f" % accu])))

                # attribute accuracy
                accu = np.mean([scores['ftt_clf_%s_%s_%s' % (data_type, attr, new_label)] for new_label in labels])
                scores['ftt_clf_%s_%s' % (data_type, attr)] = accu
                logger.info(PATTERN2.format(data_type, attr, accu))

                # log attribute confusion matrix
                logger.info("Confusion matrix for %s:" % attr)
                logger.info(confusion_ftt[attr_id])

            # overall accuracy
            accuracy = np.mean([scores['ftt_clf_%s_%s' % (data_type, a)] for a in params.attributes])
            scores['ftt_clf_%s' % data_type] = accuracy
            logger.info(PATTERN1.format(data_type, accuracy))

        # return hypothesis for fast back-translation evaluation
        return hypothesis

    def eval_back(self, data_type, scores, hypothesis):
        """
        Compute attr_1 -> attr_k -> attr_1 perplexity and BLEU scores.
        """
        logger.info("Evaluating back-translation perplexity and BLEU (%s) ..." % data_type)
        assert data_type in ['valid', 'test']
        self.encoder.eval()
        self.decoder.eval()
        params = self.params

        offset = 0
        # for each attribute
        for attr_id, attr in enumerate(params.attributes):

            # number of labels for this attribute
            n_attr = len(params.attr_values[attr])

            # for each label
            for label_id, label in enumerate(params.attr_values[attr]):

                # try all labels
                for new_label_id, new_label in enumerate(params.attr_values[attr]):

                    orig_sent = list(self.get_iterator(data_type, (attr_id, label_id)))
                    inter_sent = hypothesis[attr][(label, new_label)]
                    assert len(orig_sent) == len(inter_sent)
                    assert all([x[0].size(1) == x[1].size(0) == y[0].size(1) == y[1].size(0) for x, y in zip(orig_sent, inter_sent)])

                    hypothesis[attr][(label, new_label, label)] = []

                    # for all sentences with this label
                    for (sent1, len1, attr1), (sent2, len2, attr2) in zip(orig_sent, inter_sent):

                        # sanity check
                        assert sent1.size(1) == sent2.size(1) == len1.size(0) == len2.size(0)
                        assert (attr1[:, attr_id] - offset == label_id).sum().item() == attr1.size(0)
                        assert (attr2[:, attr_id] - offset == new_label_id).sum().item() == attr2.size(0)

                        #  cuda batch / encode sentence
                        sent1, attr1 = sent1.cuda(), attr1.cuda()
                        sent2, attr2 = sent2.cuda(), attr2.cuda()
                        encoded = self.encoder(sent2, len2)

                        # update attribute / generate hypothesis with new attributes
                        max_len = int(1.5 * len2.max() + 10)
                        sent3, len3, _ = self.decoder.generate(encoded, attr1, max_len=max_len)

                        # save hypothesis
                        hypothesis[attr][(label, new_label, label)].append((sent3, len3, attr1.clone()))

            offset += n_attr

        #
        # export references / hypothesis - compute self BLEU
        #
        PATTERN1 = 'BLEU - {:>5}: {:.3f}'
        PATTERN2 = 'BLEU - {:>5} - {:>10}: {:.3f}'
        PATTERN3 = 'BLEU - {:>5} - {:>10} - {:>10} -> {}'

        # for each attribute
        for attr in params.attributes:
            labels = params.attr_values[attr]

            # for each label
            for label_id, label in enumerate(labels):

                # for each new label
                for new_label_id, new_label in enumerate(labels):

                    # convert hypothesis to text
                    txt = []
                    for sent, lengths, _ in hypothesis[attr][(label, new_label, label)]:
                        txt.extend(convert_to_text(sent, lengths, self.dico, params))

                    # export hypothesis / restore BPE segmentation
                    filename = 'hyp.%s.%s.%s.%s.%s.%i' % (data_type, attr, label, new_label, label, scores['epoch'])
                    hyp_path = os.path.join(params.hyp_path, filename)
                    with open(hyp_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(txt) + '\n')
                    restore_segmentation(hyp_path)

                    # new label self BLEU
                    filename = 'ref.%s.%s.%s' % (data_type, attr, label)
                    ref_path = os.path.join(params.hyp_path, filename)
                    bleu = self.eval_moses_bleu(ref_path, hyp_path)
                    scores['back_bleu_%s_%s_%s_%s_%s' % (data_type, attr, label, new_label, label)] = bleu

                # label self BLEU
                bleus = [scores['back_bleu_%s_%s_%s_%s_%s' % (data_type, attr, label, new_label, label)] for new_label in labels]
                bleu = np.mean(bleus)
                scores['back_bleu_%s_%s_%s' % (data_type, attr, label)] = bleu
                if label_id == 0:
                    logger.info(PATTERN3.format(data_type, attr, '', " | ".join(["%10s" % l for l in labels + ['Total']])))
                logger.info(PATTERN3.format(data_type, attr, label, " | ".join(["%10.2f" % b for b in bleus] + ["%10.2f" % bleu])))

            # attribute self BLEU
            bleu = np.mean([scores['back_bleu_%s_%s_%s' % (data_type, attr, label)] for label in labels])
            scores['back_bleu_%s_%s' % (data_type, attr)] = bleu
            logger.info(PATTERN2.format(data_type, attr, bleu))

        # overall self BLEU
        bleu = np.mean([scores['back_bleu_%s_%s' % (data_type, attr)] for attr in params.attributes])
        scores['back_bleu_%s' % data_type] = bleu
        logger.info(PATTERN1.format(data_type, bleu))

    def eval_para(self, scores):
        """
        Evaluate with parallel sentences.
        """
        params = self.params

        assert not (params.eval_para == '') ^ ('eval' not in self.data)
        if 'eval' not in self.data:
            return

        for name, (src_data, ref_data) in self.data['eval'].items():

            src_sent = [x[:2] for x in src_data.get_iterator(shuffle=False, group_by_size=False)()]
            tgt_attr = [x[2] for x in ref_data[0].get_iterator(shuffle=False, group_by_size=False)()]

            assert len(src_sent) == len(tgt_attr)
            assert all(sent.size(1) == len(lengths) == len(attr) for (sent, lengths), attr in zip(src_sent, tgt_attr))

            txt = []
            for (sent1, len1), attr in zip(src_sent, tgt_attr):
                sent1, attr = sent1.cuda(), attr.cuda()
                encoded = self.encoder(sent1, len1)
                max_len = int(1.5 * len1.max() + 10)
                sent2, len2, _ = self.decoder.generate(encoded, attr, max_len=max_len)
                txt.extend(convert_to_text(sent2, len2, self.dico, params))

            # export sentences
            filename = 'eval.%s.hyp.%i' % (name, scores['epoch'])
            hyp_path = os.path.join(params.hyp_path, filename)
            with open(hyp_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(txt) + '\n')

            # restore BPE segmentation
            restore_segmentation(hyp_path)

            # compute BLEU
            ref_path = os.path.join(params.hyp_path, 'eval.%s.ref' % name)
            bleu = self.eval_moses_bleu(ref_path, hyp_path)
            scores['eval_para_%s' % name] = bleu
            logger.info("Parallel sentences BLEU - %s: %.5f" % (name, bleu))

        # overall BLEU
        bleu = np.mean([scores['eval_para_%s' % name] for name in self.data['eval'].keys()])
        scores['eval_para'] = bleu
        logger.info("Parallel sentences BLEU: %.5f" % bleu)

    def run_all_evals_(self, epoch):
        """
        Run all evaluations.
        """
        scores = OrderedDict({'epoch': epoch})

        with torch.no_grad():

            for data_type in ['valid', 'test']:
                hypothesis = self.eval_swap_bleu_clf(data_type, scores)
                # self.eval_back(data_type, scores, hypothesis)
            self.eval_para(scores)

        return scores


class EvaluatorClassifier(EvaluatorBase):

    def __init__(self, trainer, data, params):
        """
        Initialize evaluator.
        """
        self.classifier = trainer.classifier
        self.data = data
        self.params = params

    def eval_clf(self, data_type, scores):
        """
        Evaluate the classifier.
        """
        logger.info("Evaluating classifier (%s) ..." % data_type)
        assert data_type in ['valid', 'test']
        self.classifier.eval()

        params = self.params

        # initialize confusion matrices
        confusion = []
        for a in params.attributes:
            n_attr = len(params.attr_values[a])
            confusion.append(np.zeros((n_attr, n_attr), dtype=np.int32))

        for (sent, lengths, attr) in self.get_iterator(data_type):

            # classify
            clf_scores = self.classifier(sent.cuda(), lengths)

            # look at each attribute predictions
            offset = 0
            for i, a in enumerate(self.params.attributes):
                n = len(self.params.attr_values[a])
                predictions = clf_scores[:, offset:offset + n].cpu().numpy().argmax(1)
                for j, p in enumerate(predictions):
                    confusion[i][attr[j, i] - offset, p] += 1
                offset += n

        PATTERN1 = 'Accuracy - {:>5}: {:.3f}'
        PATTERN2 = 'Accuracy - {:>5} - {:>10}: {:.3f}'
        PATTERN3 = 'Accuracy - {:>5} - {:>10} - {:>4}: {:.3f}'

        # accuracy for each attribute
        for i, a in enumerate(self.params.attributes):

            # for each label
            for j, l in enumerate(self.params.attr_values[a]):
                accuracy = 100 * float(confusion[i][j, j]) / float(confusion[i][j].sum())
                scores['clf_%s_%s_%s' % (a, l, data_type)] = accuracy
                logger.info(PATTERN3.format(data_type, a, l, accuracy))

            # overall attribute accuracy
            accuracy = 100 * float(confusion[i].trace()) / float(confusion[i].sum())
            scores['clf_%s_%s' % (a, data_type)] = accuracy
            logger.info(PATTERN2.format(data_type, a, accuracy))

            # log confusion matrix
            logger.info("Confusion matrix for %s:" % a)
            logger.info(confusion[i])

        # overall accuracy
        accuracy = 100 * float(sum([x.trace() for x in confusion])) / float(sum([x.sum() for x in confusion]))
        scores['clf_%s' % data_type] = accuracy
        logger.info(PATTERN1.format(data_type, accuracy))

    def run_all_evals_(self, epoch):
        """
        Run all evaluations.
        """
        scores = OrderedDict({'epoch': epoch})

        with torch.no_grad():
            for data_type in ['valid', 'test']:
                self.eval_clf(data_type, scores)

        return scores


class EvaluatorLM(EvaluatorBase):

    def __init__(self, trainer, data, params):
        """
        Initialize evaluator.
        """
        self.lm = trainer.lm
        self.data = data
        self.params = params

    def eval_lm(self, data_type, scores):
        """
        Evaluate the language model.
        """
        logger.info("Evaluating language model (%s) ..." % data_type)
        assert data_type in ['valid', 'test']
        self.lm.eval()

        total_loss = 0
        total_words = 0

        for (sent, lengths, attr) in self.get_iterator(data_type):

            sent, attr = sent.cuda(), attr.cuda()
            log_probs = self.lm(sent[:-1], lengths - 1, attr)
            total_loss += F.cross_entropy(log_probs.view(-1, self.params.n_words), sent[1:].view(-1), size_average=False)
            total_words += (lengths - 1).sum()

        perplexity = np.exp(total_loss.item() / total_words.item())
        scores['lm_ppl_%s' % data_type] = perplexity
        logger.info("Language model perplexity (%s): %.3f" % (data_type, perplexity))

    def eval_attr_lm(self, data_type, scores):
        """
        Evaluate the language model with attributes.
        """
        logger.info("Evaluating language model with attributes (%s) ..." % data_type)
        assert data_type in ['valid', 'test']
        self.lm.eval()

        offset = 0

        # for each attribute
        for attr_id, a in enumerate(self.params.attributes):

            # for each label
            for label_id, label in enumerate(self.params.attr_values[a]):

                # try all labels
                for new_label_id, new_label in enumerate(self.params.attr_values[a]):
                    
                    total_loss = 0
                    total_words = 0

                    for (sent, lengths, attr) in self.get_iterator(data_type, (attr_id, label_id)):

                        # check and update attribute
                        assert (attr[:, attr_id] - offset == label_id).sum() == attr.size(0)
                        attr[:, attr_id] = new_label_id + offset

                        sent, attr = sent.cuda(), attr.cuda()
                        log_probs = self.lm(sent[:-1], lengths - 1, attr)
                        total_loss += F.cross_entropy(
                            log_probs.view(-1, self.params.n_words),
                            sent[1:].view(-1),
                            size_average=False
                        )
                        total_words += (lengths - 1).sum()

                    perplexity = np.exp(total_loss.item() / total_words.item())
                    scores['lm_ppl_%s_%s_%s_%s' % (a, label, new_label, data_type)] = perplexity
                    logger.info("Language model perplexity (%s) %s - %s - %s: %.3f"
                                % (data_type, a, label, new_label, perplexity))

            offset += len(self.params.attr_values[a])

    def run_all_evals_(self, epoch):
        """
        Run all evaluations.
        """
        scores = OrderedDict({'epoch': epoch})

        with torch.no_grad():
            for data_type in ['valid', 'test']:
                self.eval_lm(data_type, scores)
                self.eval_attr_lm(data_type, scores)

        return scores

def convert_to_text(batch, lengths, dico, params):
    """
    Convert a batch of sentences to a list of text sentences.
    """
    batch = batch.cpu().numpy()
    lengths = lengths.cpu().numpy()
    bos_index = params.bos_index

    slen, bs = batch.shape
    assert lengths.max() == slen and lengths.shape[0] == bs
    assert (batch[0] == bos_index).sum() == bs
    assert (batch == params.eos_index).sum() == bs

    sentences = []
    for j in range(bs):
        words = []
        for k in range(1, lengths[j]):
            if batch[k, j] == params.eos_index:
                break
            words.append(dico[batch[k, j]])
        sentences.append(" ".join(words))
    return sentences
