# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from .model import LSTM_PARAMS


hashs = {}


def assert_equal(x, y):
    assert x.size() == y.size()
    assert (x.data - y.data).abs().sum() == 0


def hash_data(x):
    """
    Compute a hash on tensor data.
    """
    # TODO make a better hash function (although this is good enough for embeddings)
    return (x.data.sum(), x.data.abs().sum())


def test_sharing(encoder, decoder, params):
    """
    Test parameters sharing between the encoder,
    the decoder, and the language model.
    Test that frozen parameters are not being updated.
    """
    if not params.attention:  # TODO
        return
    assert params.attention is True

    # frozen parameters
    if params.freeze_enc_emb:
        k = 'enc_emb'
        if k in hashs:
            assert hash_data(encoder.embeddings.weight) == hashs[k]
        else:
            hashs[k] = hash_data(encoder.embeddings.weight)
    if params.freeze_dec_emb:
        k = 'dec_emb'
        if k in hashs:
            assert hash_data(decoder.embeddings.weight) == hashs[k]
        else:
            hashs[k] = hash_data(decoder.embeddings.weight)

    #
    # decoder
    #
    # embedding layers
    if params.share_encdec_emb:
        assert_equal(encoder.embeddings.weight, decoder.embeddings.weight)
    # projection layers
    if params.share_decpro_emb:
        assert_equal(decoder.proj.weight, decoder.embeddings.weight)
