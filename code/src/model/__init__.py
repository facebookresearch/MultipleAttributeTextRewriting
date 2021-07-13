# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from collections import namedtuple


LSTM_PARAMS = ['weight_ih_l%i', 'weight_hh_l%i', 'bias_ih_l%i', 'bias_hh_l%i']
BILSTM_PARAMS = LSTM_PARAMS + ['%s_reverse' % x for x in LSTM_PARAMS]

LatentState = namedtuple('LatentState', 'dec_input, dis_input, input_len')


def check_ae_model_params(params):
    """
    Check models parameters.
    """

    # shared layers
    assert 0 <= params.dropout < 1
    assert not params.share_decpro_emb or params.lstm_proj or getattr(params, 'transformer', False) or params.emb_dim == params.hidden_dim
    assert not params.lstm_proj or not (params.attention and params.transformer)

    # attention model
    if params.attention:
        assert params.transformer or params.n_dec_layers > 1 or params.n_dec_layers == 1 and params.input_feeding
        assert params.transformer is False or params.emb_dim % params.encoder_attention_heads == 0
        assert params.transformer is False or params.emb_dim % params.decoder_attention_heads == 0
    # seq2seq model
    else:
        assert params.enc_dim == params.hidden_dim or not params.init_encoded
        assert params.enc_dim == params.hidden_dim or params.proj_mode == 'proj'
        assert params.proj_mode in ['proj', 'pool', 'last']

    # pooling on latent representation
    if params.pool_latent == '':
        params.pool_latent = None
    else:
        assert params.attention and not params.transformer
        s = params.pool_latent.split(',')
        assert len(s) == 2 and s[0] in ['max', 'avg'] and s[1].isdigit() and int(s[1]) > 1
        params.pool_latent = (s[0], int(s[1]))

    # language model
    if params.lm_before == params.lm_after == 0:
        assert params.lambda_lm == '0'
    else:
        assert params.lambda_lm != '0'

    # pretrained embeddings / freeze embeddings
    if params.pretrained_emb == '':
        assert not params.freeze_enc_emb or params.reload_enc
        assert not params.freeze_dec_emb or params.reload_dec
        assert not params.pretrained_out
    else:
        assert os.path.isfile(params.pretrained_emb)
        if params.share_encdec_emb:
            assert params.freeze_enc_emb == params.freeze_dec_emb
        else:
            assert not (params.freeze_enc_emb and params.freeze_dec_emb)
        assert not (params.share_decpro_emb and params.freeze_dec_emb)
        assert not (params.share_decpro_emb and not params.pretrained_out)
        assert not params.pretrained_out or params.lstm_proj or getattr(params, 'transformer', False) or params.emb_dim == params.hidden_dim

    # discriminator parameters
    assert params.dis_layers >= 0
    assert params.dis_hidden_dim > 0
    assert 0 <= params.dis_dropout < 1
    assert params.dis_clip >= 0

    # reload MT model
    assert params.reload_model == '' or os.path.isfile(params.reload_model)
    assert not (params.reload_model != '') ^ (params.reload_enc or params.reload_dec or params.reload_dis)

    # reload pretrained models for evaluation
    if params.eval_ftt_clf != '':
        s = [x.split(':') for x in params.eval_ftt_clf.split(',')]
        assert all([len(x) == 2 for x in s])
        assert len(s) == len(params.attributes)
        assert set([attr for attr, _ in s]) == set(params.attributes)
        assert all([attr.startswith('length_') and path == '' or os.path.isfile(path) for attr, path in s])
        params.eval_ftt_clf = {attr: path for attr, path in s}
    assert params.eval_cnn_clf == '' or os.path.isfile(params.eval_cnn_clf)
    assert params.eval_lm == '' or os.path.isfile(params.eval_lm)


def build_ae_model(params, data, cuda=True):
    """
    Build the autoencoder model.
    """
    if params.attention:
        from .attention import build_attention_model
        return build_attention_model(params, data, cuda=cuda)
    else:
        from .seq2seq import build_seq2seq_model
        return build_seq2seq_model(params, data, cuda=cuda)
