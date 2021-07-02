# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
from functools import reduce
from operator import mul
import torch
from torch import nn
from torch.nn import functional as F

from . import LatentState
from .discriminator import MultiAttrDiscriminator
from .lm import LM
from .pretrain_embeddings import initialize_embeddings
from ..utils import get_mask, reload_model


logger = getLogger()


def get_init_state(n_dec_layers, batch_size, hidden_dim, init_state=None):
    """
    Build an initial LSTM state, with optional non-zero first layer.
    """
    init = torch.cuda.FloatTensor(n_dec_layers, batch_size, hidden_dim).zero_()
    h_0 = init.clone()
    c_0 = init.clone()
    if init_state is not None:
        assert init_state.size() == (batch_size, hidden_dim)
        h_0[0] = init_state
    return (h_0, c_0)


class Encoder(nn.Module):

    ENC_ATTR = ['n_words', 'emb_dim', 'hidden_dim', 'dropout', 'n_enc_layers', 'enc_dim', 'proj_mode', 'pad_index']

    def __init__(self, params):
        """
        Encoder initialization.
        """
        super(Encoder, self).__init__()

        # model parameters
        self.n_words = params.n_words
        self.emb_dim = params.emb_dim
        self.hidden_dim = params.hidden_dim
        self.dropout = params.dropout
        self.n_enc_layers = params.n_enc_layers
        self.enc_dim = params.enc_dim
        self.proj_mode = params.proj_mode
        self.pad_index = params.pad_index
        self.freeze_enc_emb = params.freeze_enc_emb

        # embedding layers
        self.embeddings = nn.Embedding(self.n_words, self.emb_dim, padding_idx=self.pad_index)
        nn.init.normal_(self.embeddings.weight, 0, 0.1)
        nn.init.constant_(self.embeddings.weight[self.pad_index], 0)

        # LSTM layers
        self.lstm = nn.LSTM(self.emb_dim, self.hidden_dim, num_layers=self.n_enc_layers, dropout=self.dropout)

        # projection layers
        if self.proj_mode == 'proj':
            self.proj = nn.Linear(self.hidden_dim, self.enc_dim)
        else:
            self.proj = None

    def forward(self, x, lengths):
        """
        Input:
            - LongTensor of size (slen, bs), word indices
            - LongTensor of size (bs,), sentence lengths
        Output:
            - FloatTensor of size (bs, enc_dim),
              representing the encoded state of each sentence
        """
        is_cuda = x.is_cuda

        # embeddings
        slen, bs = x.size(0), x.size(1)
        if x.dim() == 2:
            embeddings = self.embeddings(x)
        else:
            assert x.dim() == 3 and x.size(2) == self.n_words
            embeddings = x.view(slen * bs, -1).mm(self.embeddings.weight).view(slen, bs, self.emb_dim)
        embeddings = embeddings.detach() if self.freeze_enc_emb else embeddings
        embeddings = F.dropout(embeddings, p=self.dropout, training=self.training)
        assert lengths.max() == slen and lengths.size(0) == bs
        assert embeddings.size() == (slen, bs, self.emb_dim)

        # LSTM
        lstm_output, (_, _) = self.lstm(embeddings)
        assert lstm_output.size() == (slen, bs, self.hidden_dim)

        # encoded sentences representation
        if self.proj_mode == 'pool':
            latent_state = lstm_output.max(0)[0]
        else:
            # select the last state of each sentence
            mask = get_mask(lengths, False, expand=self.hidden_dim, batch_first=True, cuda=is_cuda)
            h_t = lstm_output.transpose(0, 1).masked_select(mask).view(bs, self.hidden_dim)
            if self.proj_mode == 'proj':
                latent_state = self.proj(h_t)
            elif self.proj_mode == 'last':
                latent_state = h_t

        return LatentState(input_len=lengths, dec_input=latent_state, dis_input=latent_state)


class Decoder(nn.Module):

    DEC_ATTR = ['attributes', 'attr_values', 'n_words', ('share_encdec_emb', False), ('share_decpro_emb', False), 'emb_dim', 'hidden_dim', 'dropout', 'n_dec_layers', 'enc_dim', 'init_encoded', 'eos_index', 'pad_index', 'bos_index']

    def __init__(self, params, encoder):
        """
        Decoder initialization.
        """
        super(Decoder, self).__init__()

        # model parameters
        self.attributes = params.attributes
        self.attr_values = params.attr_values
        self.n_words = params.n_words
        self.share_encdec_emb = params.share_encdec_emb
        self.share_decpro_emb = params.share_decpro_emb
        self.emb_dim = params.emb_dim
        self.hidden_dim = params.hidden_dim
        self.lstm_proj = params.lstm_proj
        self.dropout = params.dropout
        self.n_dec_layers = params.n_dec_layers
        self.enc_dim = params.enc_dim
        self.init_encoded = params.init_encoded
        self.freeze_dec_emb = params.freeze_dec_emb
        self.bos_attr = params.bos_attr
        self.bias_attr = params.bias_attr
        assert not self.share_decpro_emb or self.lstm_proj or self.emb_dim == self.hidden_dim
        assert self.enc_dim == self.hidden_dim or not self.init_encoded
        assert self.bos_attr in ['', 'avg', 'cross']
        assert self.bias_attr in ['', 'avg', 'cross']

        # indexes
        self.bos_index = params.bos_index
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index

        # attribute embeddings / bias
        if self.bos_attr != '' or self.bias_attr != '':
            self.register_buffer('attr_offset', params.attr_offset.clone())
            self.register_buffer('attr_shifts', params.attr_shifts.clone())
        if self.bos_attr != '':
            n_bos_attr = sum(params.n_labels) if self.bos_attr == 'avg' else reduce(mul, params.n_labels, 1)
            self.bos_attr_embeddings = nn.Embedding(n_bos_attr, self.emb_dim)
        if self.bias_attr != '':
            n_bias_attr = sum(params.n_labels) if self.bias_attr == 'avg' else reduce(mul, params.n_labels, 1)
            self.bias_attr_embeddings = nn.Embedding(n_bias_attr, self.n_words)

        # embedding layers
        if self.share_encdec_emb:
            logger.info("Sharing encoder and decoder input embeddings")
            self.embeddings = encoder.embeddings
        else:
            self.embeddings = nn.Embedding(self.n_words, self.emb_dim, padding_idx=self.pad_index)
            nn.init.normal_(self.embeddings.weight, 0, 0.1)
            nn.init.constant_(self.embeddings.weight[self.pad_index], 0)

        # LSTM layers
        isize = self.emb_dim + (0 if self.init_encoded else self.enc_dim)
        self.lstm = nn.LSTM(isize, self.hidden_dim, num_layers=self.n_dec_layers, dropout=self.dropout)

        # projection layers between LSTM and output embeddings
        if self.lstm_proj:
            self.lstm_proj_layer = nn.Linear(self.hidden_dim, self.emb_dim)
            proj_output_dim = self.emb_dim
        else:
            self.lstm_proj_layer = None
            proj_output_dim = self.hidden_dim

        # projection layers
        proj = nn.Linear(proj_output_dim, self.n_words)
        if self.share_decpro_emb:
            logger.info("Sharing input embeddings and projection matrix in the decoder")
            proj.weight = self.embeddings.weight
        self.proj = proj

    def get_bos_attr(self, attr):
        """
        Generate beginning of sentence attribute embedding.
        """
        if self.bos_attr == 'avg':
            return self.bos_attr_embeddings(attr).mean(1)
        if self.bos_attr == 'cross':
            return self.bos_attr_embeddings(((attr - self.attr_offset[None]) * self.attr_shifts[None]).sum(1))
        assert False

    def get_bias_attr(self, attr):
        """
        Generate attribute bias.
        """
        if self.bias_attr == 'avg':
            return self.bias_attr_embeddings(attr).mean(1)
        if self.bias_attr == 'cross':
            return self.bias_attr_embeddings(((attr - self.attr_offset[None]) * self.attr_shifts[None]).sum(1))
        assert False

    def forward(self, encoded, y, attr, one_hot=False):
        """
        Input:
            - LongTensor of size (slen, bs), word indices
              or
              LongTensor of size (slen, bs, n_words), one-hot word embeddings
            - LongTensor of size (bs,), sentence lengths
            - FloatTensor of size (bs, hidden_dim), latent
              state representing sentences
        Output:
            - FloatTensor of size (slen, bs, n_words),
              representing the score of each word in each sentence
        """
        assert encoded.input_len.size(0) == encoded.dec_input.size(0) == y.size(1)
        assert attr.size() == (encoded.input_len.size(0), len(self.attributes))
        latent = encoded.dec_input

        # embeddings
        if one_hot:
            slen, bs, _ = y.size()
            embeddings = y.view(slen * bs, self.n_words).mm(self.embeddings.weight)
            embeddings = embeddings.view(slen, bs, self.emb_dim)
        else:
            slen, bs = y.size()
            embeddings = self.embeddings(y)
        embeddings = embeddings.detach() if self.freeze_dec_emb else embeddings
        if self.bos_attr != '':
            embeddings[0] = self.get_bos_attr(attr)
        embeddings = F.dropout(embeddings, p=self.dropout, training=self.training)
        assert latent.size() == (bs, self.enc_dim)
        assert embeddings.size() == (slen, bs, self.emb_dim)

        if self.init_encoded:
            init = get_init_state(self.n_dec_layers, bs, self.hidden_dim, latent)
            lstm_input = embeddings
        else:
            init = None
            encoded = latent.unsqueeze(0).expand(slen, bs, self.enc_dim)
            lstm_input = torch.cat([embeddings, encoded], 2)

        # LSTM
        lstm_output, (_, _) = self.lstm(lstm_input, init)
        assert lstm_output.size() == (slen, bs, self.hidden_dim)

        # word scores
        output = F.dropout(lstm_output, p=self.dropout, training=self.training).view(-1, self.hidden_dim)
        if self.lstm_proj_layer is not None:
            output = F.relu(self.lstm_proj_layer(output))
        scores = self.proj(output).view(slen, bs, self.n_words)
        if self.bias_attr != '':
            scores = scores + self.get_bias_attr(attr)[None]
        return scores

    def generate(self, encoded, attr, max_len=200, sample=False, temperature=None):
        """
        Generate a sentence from a given initial state.
        Input:
            - FloatTensor of size (batch_size, hidden_dim) representing
              sentences encoded in the latent space
        Output:
            - LongTensor of size (seq_len, batch_size), word indices
            - LongTensor of size (batch_size,), sentence lengths
        """
        assert encoded.input_len.size(0) == encoded.dec_input.size(0)
        assert attr.size() == (encoded.input_len.size(0), len(self.attributes))
        latent = encoded.dec_input
        is_cuda = latent.is_cuda
        assert (sample is True) ^ (temperature is None)
        one_hot = None  # [] if temperature is not None else None

        # initialize generated sentences batch
        bs = latent.size(0)
        cur_len = 1
        if self.init_encoded:
            h_c = get_init_state(self.n_dec_layers, bs, self.hidden_dim, latent)
        else:
            h_c = None
        decoded = torch.LongTensor(max_len, bs).fill_(self.pad_index)
        decoded = decoded.cuda() if is_cuda else decoded
        decoded[0] = self.bos_index

        # decoding
        while cur_len < max_len:
            # previous word embeddings
            if cur_len == 1 and self.bos_attr != '':
                embeddings = self.get_bos_attr(attr)
            else:
                embeddings = self.embeddings(decoded[cur_len - 1])
            embeddings = F.dropout(embeddings, p=self.dropout, training=self.training)
            if not self.init_encoded:
                embeddings = torch.cat([embeddings, latent], 1)
            lstm_output, h_c = self.lstm(embeddings.unsqueeze(0), h_c)
            output = F.dropout(lstm_output, p=self.dropout, training=self.training).view(bs, self.hidden_dim)
            if self.lstm_proj_layer is not None:
                output = F.relu(self.lstm_proj_layer(output))
            scores = self.proj(output)
            if self.bias_attr != '':
                scores = scores + self.get_bias_attr(attr)
            scores = scores.data
            assert scores.size() == (bs, self.n_words)

            # select next words: sample (Gumbel Softmax) or one-hot
            if sample:
                # if temperature is not None:
                #     gumbel = gumbel_softmax(scores, temperature, hard=True)
                #     next_words = gumbel.max(1)[1]
                #     one_hot.append(gumbel)
                # else:
                next_words = torch.multinomial(F.softmax(scores / temperature, dim=1), 1).squeeze(1)
            else:
                next_words = scores.max(1)[1]
            assert next_words.size() == (bs,)
            decoded[cur_len] = next_words
            cur_len += 1

            # stop when there is a </s> in each sentence
            if decoded.eq(self.eos_index).sum(0).ne(0).sum() == bs:
                break

        # compute the length of each generated sentence, and
        # put some padding after the end of each sentence
        lengths = torch.LongTensor(bs).fill_(cur_len)
        for i in range(bs):
            for j in range(cur_len):
                if decoded[j, i] == self.eos_index:
                    if j + 1 < max_len:
                        decoded[j + 1:, i] = self.pad_index
                    lengths[i] = j + 1
                    break
            if lengths[i] == max_len:
                decoded[-1, i] = self.eos_index

        if one_hot is not None:
            one_hot = torch.cat([x.unsqueeze(0) for x in one_hot], 0)
            assert one_hot.size() == (cur_len - 1, bs, self.n_words)
        return decoded[:cur_len], lengths, one_hot


def build_seq2seq_model(params, data, cuda=True):
    """
    Build a encoder / decoder, and the decoder reconstruction loss function.
    """
    # encoder / decoder / discriminator
    logger.info("============ Building seq2seq model - Encoder ...")
    encoder = Encoder(params)
    logger.info("")
    logger.info("============ Building seq2seq model - Decoder ...")
    decoder = Decoder(params, encoder)
    logger.info("")
    if params.lambda_dis != "0":
        logger.info("============ Building seq2seq model - Discriminator ...")
        discriminator = MultiAttrDiscriminator(params)
        logger.info("")
    else:
        discriminator = None

    # loss function for decoder reconstruction
    loss_weight = torch.FloatTensor(params.n_words).fill_(1)
    loss_weight[params.pad_index] = 0
    decoder.loss_fn = nn.CrossEntropyLoss(loss_weight, size_average=True)

    # language model
    if params.lambda_lm != "0":
        logger.info("============ Building seq2seq model - Language model ...")
        lm = LM(params, data['dico'])
        logger.info("")
    else:
        lm = None

    # cuda - models on CPU will be synchronized and don't need to be reloaded
    if cuda:
        encoder.cuda()
        decoder.cuda()
        if discriminator is not None:
            discriminator.cuda()
        if lm is not None:
            lm.cuda()

        # initialize the model with pretrained embeddings
        assert not (getattr(params, 'cpu_thread', False)) ^ (data is None)
        if data is not None:
            initialize_embeddings(encoder, decoder, params, data)

        # reload encoder / decoder / discriminator
        if params.reload_model != '':
            assert os.path.isfile(params.reload_model)
            logger.info("Reloading model from %s ..." % params.reload_model)
            reloaded = torch.load(params.reload_model)
            if params.reload_enc:
                logger.info("Reloading encoder...")
                reload_model(encoder, reloaded['enc'], encoder.ENC_ATTR)
            if params.reload_dec:
                logger.info("Reloading decoder...")
                reload_model(decoder, reloaded['dec'], decoder.DEC_ATTR)
            if params.reload_dis:
                logger.info("Reloading discriminator...")
                reload_model(discriminator, reloaded['dis'], discriminator.DIS_ATTR)

    # log models
    logger.info("============ Model summary")
    logger.info("Encoder: {}".format(encoder))
    logger.info("Decoder: {}".format(decoder))
    logger.info("Discriminator: {}".format(discriminator))
    logger.info("LM: {}".format(lm))
    logger.info("")

    return encoder, decoder, discriminator, lm
