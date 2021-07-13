# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import numpy as np
import torch

from .dataset import Dataset
from .dictionary import BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, SPECIAL_WORD


logger = getLogger()


loaded_data = {}  # store binarized datasets in memory in case of multiple reloadings


def load_binarized(path, params):
    """
    Load a binarized dataset and log main statistics.
    """
    if path in loaded_data:
        logger.info("Reloading data loaded from %s ..." % path)
        return loaded_data[path]
    assert os.path.isfile(path), path
    logger.info("Loading data from %s ..." % path)
    data = torch.load(path)
    data['positions'] = data['positions'].numpy()
    logger.info("%i words (%i unique) in %i sentences with %i attributes. %i unknown words (%i unique)." % (
        len(data['sentences']) - len(data['positions']),
        len(data['dico']), len(data['positions']),
        len(data['attr_values']),
        sum(data['unk_words'].values()), len(data['unk_words'])
    ))
    # add length attribute if required
    len_attrs = [attr for attr in params.attributes if attr.startswith('length_')]
    assert len(len_attrs) <= 1
    if len(len_attrs) == 1:
        len_attr = len_attrs[0]
        assert len_attr[len('length_'):].isdigit()
        bs = int(len_attr[len('length_'):])
        lm = params.max_len
        assert bs >= 1 and lm >= 1 and len_attr not in data['attr_values']
        sr = np.arange(0, lm + 1 if lm % bs == 0 else lm + bs - lm % bs + 1, bs)
        len_labels = np.ceil((data['positions'][:, 1] - data['positions'][:, 0]).astype(np.float32) / bs) - 1
        len_labels = np.minimum(len_labels.astype(np.int64), len(sr) - 2)
        assert len_labels.min() >= 0
        data['attr_values'][len_attr] = ['%s-%s' % (sr[i], sr[i + 1]) for i in range(len(sr) - 1)]
        params.size_ranges = sr
        params.bucket_size = bs
    else:
        len_attr = None
    # maximum vocabulary size
    if params.max_vocab != -1:
        assert params.max_vocab > 0
        logger.info("Selecting %i most frequent words ..." % params.max_vocab)
        data['dico'].prune(params.max_vocab)
        data['sentences'].masked_fill_((data['sentences'] >= params.max_vocab), data['dico'].index(UNK_WORD))
        unk_count = (data['sentences'] == data['dico'].index(UNK_WORD)).sum()
        logger.info("Now %i unknown words covering %.2f%% of the data." % (
            unk_count, 100. * unk_count / (len(data['sentences']) - len(data['positions']))
        ))
    # select relevant attributes
    assert data['attributes'].size() == (len(data['positions']), len(data['attr_values']) - len(len_attrs))
    assert all(x in data['attr_values'] for x in params.attributes)
    attr_idx = [sorted(data['attr_values'].keys()).index(x) for x in params.attributes if x != len_attr]
    data['attributes'] = data['attributes'][:, attr_idx]
    if len_attr is not None:
        data['attributes'] = torch.cat([data['attributes'], torch.from_numpy(len_labels[:, None])], 1)
    # save data to avoid identical reloading
    loaded_data[path] = data
    return data


def set_parameters(params, dico, attr_values):
    """
    Define parameters, check dictionaries / attributes.
    """
    # dictionary
    bos_index = dico.index(BOS_WORD)
    eos_index = dico.index(EOS_WORD)
    pad_index = dico.index(PAD_WORD)
    unk_index = dico.index(UNK_WORD)
    blank_index = dico.index(SPECIAL_WORD % 0)
    if hasattr(params, 'bos_index'):
        assert params.bos_index == bos_index
        assert params.eos_index == eos_index
        assert params.pad_index == pad_index
        assert params.unk_index == unk_index
        assert params.blank_index == blank_index
    else:
        params.bos_index = bos_index
        params.eos_index = eos_index
        params.pad_index = pad_index
        params.unk_index = unk_index
        params.blank_index = blank_index

    # attributes
    id2label_ = {a: {i: l for i, l in enumerate(attr_values[a])} for a in sorted(attr_values.keys()) if a in params.attributes}
    label2id_ = {a: {l: i for i, l in enumerate(attr_values[a])} for a in sorted(attr_values.keys()) if a in params.attributes}
    assert not hasattr(params, 'id2label') or params.id2label == id2label_
    assert not hasattr(params, 'label2id') or params.label2id == label2id_
    params.id2label = id2label_
    params.label2id = label2id_

    # attribute values
    attr_values_ = {k: v for k, v in attr_values.items() if k in params.attributes}
    assert not hasattr(params, 'attr_values') or params.attr_values == attr_values_
    params.attr_values = attr_values_


def load_mono_data(params, data):
    """
    Load monolingual data.
    """
    assert params.n_mono != 0

    logger.info('============ Monolingual data')

    datasets = []

    for name, path in zip(['train', 'valid', 'test'], params.mono_dataset):

        # load data
        mono_data = load_binarized(path, params)
        set_parameters(params, mono_data['dico'], mono_data['attr_values'])

        # set / check dictionary
        if 'dico' not in data:
            data['dico'] = mono_data['dico']
        else:
            assert data['dico'] == mono_data['dico']

        # monolingual data
        mono_data = Dataset(mono_data['sentences'], mono_data['attributes'],
                            mono_data['positions'], data['dico'], params)

        # remove too long sentences (train / valid only, test must remain unchanged)
        # if name != 'test':  # TODOO: check this
        mono_data.remove_long_sentences(params.max_len)

        # select a subset of sentences
        if name == 'train' and params.n_mono != -1:
            mono_data.select_data(0, params.n_mono)

        # create attribute indexes
        mono_data.create_attr_idx()

        datasets.append((name, mono_data))

    data['mono'] = {k: v for k, v in datasets}

    logger.info('')


def load_eval_data(params, data):
    """
    Load evaluation data.
    """
    if params.eval_para == '':
        return

    logger.info('============ Evaluation data')

    datasets = {}

    for name, (src_path, ref_paths) in params.eval_para.items():

        all_paths = [src_path] + ref_paths

        # load data
        all_data = []
        for path in all_paths:
            _data = load_binarized(path, params)
            set_parameters(params, _data['dico'], _data['attr_values'])
            all_data.append(_data)

        # check dictionary
        assert all([data['dico'] == __data['dico'] for __data in all_data])

        # data / check number of sentences / check same reference attributes
        all_data = [Dataset(__data['sentences'], __data['attributes'],
                            __data['positions'], data['dico'], params)
                    for __data in all_data]
        assert len(set([len(__data) for __data in all_data]))
        assert all((all_data[1].attr == __data.attr).long().sum().item() == len(all_data[1].attr)
                   for __data in all_data[2:])

        # create attribute indexes
        for _data in all_data:
            _data.create_attr_idx()

        datasets[name] = (all_data[0], all_data[1:])

    data['eval'] = datasets

    logger.info('')


def check_all_data_params(params):
    """
    Check datasets parameters.
    """
    # check attributes
    assert sorted(params.attributes) == params.attributes

    # check monolingual datasets
    assert params.n_mono != 0
    params.mono_dataset = params.mono_dataset.split(',')
    assert len(params.mono_dataset) == 3
    assert all(os.path.isfile(path) for path in params.mono_dataset)

    # check parallel evaluation data
    if not params.eval_para == '':
        params.eval_para = [x.split(':') for x in params.eval_para.split(';')]
        assert all(len(x) == 3 for x in params.eval_para)
        assert len(params.eval_para) == len(set([x for x, _, _ in params.eval_para]))
        params.eval_para = {name: (src, refs.split(',')) for name, src, refs in params.eval_para}
        assert all(os.path.isfile(src) for src, _ in params.eval_para.values())
        assert all(all(os.path.isfile(ref) for ref in refs) for _, refs in params.eval_para.values())

    # check training / lambda parameters
    assert params.train_ae or params.train_bt
    assert not (params.lambda_ae == '0') ^ (params.train_ae is False)
    assert not (params.lambda_bt == '0') ^ (params.train_bt is False)

    # check coefficients  # TODOO: update this
    # assert not (params.lambda_dis == "0") ^ (params.n_dis == 0)

    # max length / max vocab / sentence noise
    assert params.max_len > 0
    assert params.max_vocab == -1 or params.max_vocab > 0
    assert params.word_shuffle == 0 or params.word_shuffle > 1
    assert 0 <= params.word_dropout < 1
    assert 0 <= params.word_blank < 1


def check_mono_data_params(params):
    """
    Check datasets parameters.
    """
    # check attributes
    assert sorted(params.attributes) == params.attributes

    # check monolingual datasets
    assert params.n_mono != 0
    params.mono_dataset = params.mono_dataset.split(',')
    assert len(params.mono_dataset) == 3
    assert all(os.path.isfile(path) for path in params.mono_dataset)

    # max length / max vocab
    assert params.max_len > 0
    assert params.max_vocab == -1 or params.max_vocab > 0


def load_data(params, mono_only=False):
    """
    Load parallel / monolingual data.
    We start with the parallel test set, which defines the dictionaries.
    Each other dataset has to match the same dictionaries.
    The returned dictionary contains:
        - dico
        - mono (dictionary of monolingual datasets (train, valid, test))
    """
    data = {}

    # monolingual datasets
    load_mono_data(params, data)

    # evaluation datasets
    load_eval_data(params, data)

    # update parameters
    params.n_words = len(data['dico'])

    # data summary
    logger.info('============ Data summary')
    for data_type in ['train', 'valid', 'test']:
        logger.info('{: <15} - {: >5}:{: >10}'.format('Monolingual data', data_type, len(data['mono'][data_type])))
        for i, attr in enumerate(params.attributes):
            labels = params.id2label[attr]
            for j in range(len(labels)):
                count = len(data['mono'][data_type].attr_idx[i][j])
                total = len(data['mono'][data_type])
                logger.info('\t{: <9} - {: >10}:{: >8} ({:5.2f}%)'.format(
                    attr, labels[j], count, 100. * count / total
                ))

    logger.info('')
    return data
