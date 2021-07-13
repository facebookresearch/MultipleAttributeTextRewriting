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
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from . import LatentState
from .discriminator import MultiAttrDiscriminator, MultiAttrDiscriminatorLSTM
from .lm import LM
from ..modules.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyLoss
from .pretrain_embeddings import initialize_embeddings
from ..utils import get_mask, reload_model
# from ..gumbel import gumbel_softmax


logger = getLogger()


class Encoder(nn.Module):

    ENC_ATTR = ['n_words', 'emb_dim', 'hidden_dim', 'dropout', 'n_enc_layers', 'pad_index', 'pool_latent', 'dis_input_proj']

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
        self.pad_index = params.pad_index
        self.freeze_enc_emb = params.freeze_enc_emb
        self.max_len = params.max_len
        self.pool_latent = params.pool_latent
        self.dis_input_proj = params.dis_input_proj

        # embedding layers
        self.embeddings = nn.Embedding(self.n_words, self.emb_dim, padding_idx=self.pad_index)
        nn.init.normal_(self.embeddings.weight, 0, 0.1)
        nn.init.constant_(self.embeddings.weight[self.pad_index], 0)

        # LSTM layers
        self.lstm = nn.LSTM(self.emb_dim, self.hidden_dim, num_layers=self.n_enc_layers, dropout=self.dropout, bidirectional=True)

        # projection layers
        self.proj = nn.Linear(2 * self.hidden_dim, self.emb_dim, bias=False)

    def forward(self, x, lengths):
        """
        Input:
            - LongTensor of size (slen, bs), word indices
            - List of length bs, containing the sentence lengths
            Sentences have to be ordered by decreasing length
        Output:
            - FloatTensor of size (slen, bs, 2 * hidden_dim),
              representing the encoded state of each sentence
        """
        is_cuda = x.is_cuda
        sort_len = lengths.type_as(x.data).sort(0, descending=True)[1]
        sort_len_rev = sort_len.sort()[1]

        # embeddings
        slen, bs = x.size(0), x.size(1)
        if x.dim() == 2:
            embeddings = self.embeddings(x.index_select(1, sort_len))
        else:
            assert x.dim() == 3 and x.size(2) == self.n_words
            embeddings = x.view(slen * bs, -1).mm(self.embeddings.weight).view(slen, bs, self.emb_dim).index_select(1, sort_len)
        embeddings = embeddings.detach() if self.freeze_enc_emb else embeddings
        embeddings = F.dropout(embeddings, p=self.dropout, training=self.training)
        lstm_input = pack_padded_sequence(embeddings, sorted(lengths.tolist(), reverse=True))
        assert lengths.max() == slen and lengths.size(0) == bs
        assert lstm_input.data.size() == (sum(lengths), self.emb_dim)

        # LSTM
        lstm_output, (_, _) = self.lstm(lstm_input)
        assert lstm_output.data.size() == (lengths.sum(), 2 * self.hidden_dim)

        # get a padded version of the LSTM output
        padded_output, _ = pad_packed_sequence(lstm_output)
        assert padded_output.size() == (slen, bs, 2 * self.hidden_dim)

        # project biLSTM output
        padded_output = self.proj(padded_output.view(slen * bs, -1)).view(slen, bs, self.emb_dim)

        # re-order sentences in their original order
        padded_output = padded_output.index_select(1, sort_len_rev)

        # pooling on latent representation
        if self.pool_latent is not None:
            pool, ks = self.pool_latent
            p = ks - slen if slen < ks else (0 if slen % ks == 0 else ks - (slen % ks))
            y = padded_output.transpose(0, 2)
            if p > 0:
                value = 0 if pool == 'avg' else -1e9
                y = F.pad(y, (0, p), mode='constant', value=value)
            y = (F.avg_pool1d if pool == 'avg' else F.max_pool1d)(y, ks)
            padded_output = y.transpose(0, 2)
            lengths = (lengths.float() / ks).ceil().long()

        # discriminator input
        dis_input = lstm_output.data
        if self.dis_input_proj:
            mask = get_mask(lengths, all_words=True, expand=self.emb_dim, batch_first=False, cuda=is_cuda)
            dis_input = padded_output.masked_select(mask).view(lengths.sum().item(), self.emb_dim)

        return LatentState(input_len=lengths, dec_input=padded_output, dis_input=dis_input)


class Decoder(nn.Module):

    DEC_ATTR = ['attributes', 'attr_values', 'n_words', ('share_encdec_emb', False), ('share_decpro_emb', False), 'emb_dim', 'hidden_dim', 'dropout', 'n_dec_layers', 'input_feeding', 'eos_index', 'pad_index', 'bos_index']

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
        self.input_feeding = params.input_feeding
        self.freeze_dec_emb = params.freeze_dec_emb
        self.bos_attr = params.bos_attr
        self.bias_attr = params.bias_attr
        assert not self.share_decpro_emb or self.lstm_proj or self.emb_dim == self.hidden_dim
        assert self.n_dec_layers > 1 or self.n_dec_layers == 1 and self.input_feeding
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
        isize1 = self.emb_dim + (self.emb_dim if self.input_feeding else 0)
        isize2 = self.hidden_dim + (0 if self.input_feeding else self.emb_dim)
        self.lstm1 = nn.LSTM(isize1, self.hidden_dim, num_layers=1, dropout=self.dropout, bias=True)
        self.lstm2 = nn.LSTM(isize2, self.hidden_dim, num_layers=self.n_dec_layers - 1, dropout=self.dropout, bias=True) if self.n_dec_layers > 1 else None

        # attention layers
        self.att_proj = nn.Linear(self.hidden_dim, self.emb_dim)

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

    def get_attention(self, latent, h_t, y_t, mask):
        """
        Compute the attention vector for a single decoder step.
        """
        att_proj = self.att_proj
        proj_hidden = att_proj(h_t)                                                   # (bs, emb_dim)
        proj_hidden = (proj_hidden + y_t).unsqueeze(0)                                # (1, bs, emb_dim)

        att_weights = latent * proj_hidden.expand_as(latent)                          # (xlen, bs, emb_dim)
        att_weights = att_weights.sum(2)                                              # (xlen, bs)
        att_weights = att_weights.masked_fill(mask, -1e30)                            # (xlen, bs)
        att_weights = F.softmax(att_weights.transpose(0, 1), dim=-1).transpose(0, 1)  # (xlen, bs)

        att_vectors = latent * att_weights.unsqueeze(2).expand_as(latent)             # (xlen, bs, emb_dim)
        att_vectors = att_vectors.sum(0, keepdim=True)                                # (1, bs, emb_dim)

        # print " ".join("%.4f" % x for x in att_weights.data.cpu().numpy()[:, 0])
        assert att_vectors.size() == (1, proj_hidden.size(1), self.emb_dim)
        return att_vectors

    def get_full_attention(self, latent, h, y, mask):
        """
        Compute the attention vectors for all decoder steps.
        """
        latent = latent.transpose(0, 1)
        h = h.transpose(0, 1).contiguous()
        y = y.transpose(0, 1)

        bs = latent.size(0)
        xlen = latent.size(1)
        ylen = y.size(1)

        att_proj = self.att_proj
        proj_hidden = att_proj(h.view(bs * ylen, self.hidden_dim))                   # (bs * ylen, emb_dim)
        proj_hidden = proj_hidden.view(bs, ylen, self.emb_dim)                       # (bs, ylen, emb_dim)
        proj_hidden = (proj_hidden + y)                                              # (bs, ylen, emb_dim)

        att_weights = proj_hidden.bmm(latent.transpose(1, 2))                        # (bs, ylen, xlen)
        att_weights = att_weights.masked_fill(mask, -1e30)                           # (bs, ylen, xlen)
        att_weights = F.softmax(att_weights.view(bs * ylen, xlen), dim=-1)           # (bs * ylen, xlen)
        att_weights = att_weights.view(bs, ylen, xlen)                               # (bs, ylen, xlen)

        att_vectors = latent.unsqueeze(1).expand(bs, ylen, xlen, self.emb_dim)       # (bs, ylen, xlen, emb_dim)
        att_vectors = att_vectors * att_weights.unsqueeze(3).expand_as(att_vectors)  # (bs, ylen, xlen, emb_dim)
        att_vectors = att_vectors.sum(2)                                             # (bs, ylen, emb_dim)

        assert att_vectors.size() == (bs, ylen, self.emb_dim)
        return att_vectors.transpose(0, 1)

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
        latent = encoded.dec_input
        x_len = encoded.input_len
        is_cuda = latent.is_cuda

        # check inputs
        assert x_len.size(0) == y.size(1)
        assert latent.size() == (x_len.max(), x_len.size(0), self.emb_dim)
        assert attr.size() == (x_len.size(0), len(self.attributes))

        # embeddings
        if one_hot:
            y_len, bs, _ = y.size()
            embeddings = y.view(y_len * bs, self.n_words).mm(self.embeddings.weight)
            embeddings = embeddings.view(y_len, bs, self.emb_dim)
        else:
            y_len, bs = y.size()
            embeddings = self.embeddings(y)
        embeddings = embeddings.detach() if self.freeze_dec_emb else embeddings
        if self.bos_attr != '':
            embeddings[0] = self.get_bos_attr(attr)
        embeddings = F.dropout(embeddings, p=self.dropout, training=self.training)

        if self.input_feeding:
            mask = get_mask(x_len, True, cuda=is_cuda) == 0  # attention mask
            h_c = None
            hidden_states = [latent.data.new(1, bs, self.hidden_dim).zero_()]
            attention_states = []

            for i in range(y_len):
                # attention layer
                attention = self.get_attention(latent, hidden_states[-1][0], embeddings[i], mask)
                attention_states.append(attention)

                # lstm step
                lstm_input = embeddings[i:i + 1]
                lstm_input = torch.cat([lstm_input, attention], 2)
                h_t, h_c = self.lstm1(lstm_input, h_c)
                assert h_t.size() == (1, bs, self.hidden_dim)
                hidden_states.append(h_t)

            # first layer LSTM output
            lstm_output = torch.cat(hidden_states[1:], 0)
            assert lstm_output.size() == (y_len, bs, self.hidden_dim)

            # lstm (layers > 1)
            if self.n_dec_layers > 1:
                lstm_output = F.dropout(lstm_output, p=self.dropout, training=self.training)
                lstm_output, (_, _) = self.lstm2(lstm_output)
                assert lstm_output.size() == (y_len, bs, self.hidden_dim)

        else:
            # first LSTM layer
            lstm_output, (_, _) = self.lstm1(embeddings)
            assert lstm_output.size() == (y_len, bs, self.hidden_dim)

            # attention layer
            mask = get_mask(x_len, True, expand=int(y_len), batch_first=True, cuda=is_cuda).transpose(1, 2) == 0
            att_input = torch.cat([latent.data.new(1, bs, self.hidden_dim).zero_(), lstm_output[:-1]], 0)
            attention = self.get_full_attention(latent, att_input, embeddings, mask)
            assert attention.size() == (y_len, bs, self.emb_dim)

            # > 1 LSTM layers
            lstm_output = F.dropout(lstm_output, p=self.dropout, training=self.training)
            lstm_output = torch.cat([lstm_output, attention], 2)
            lstm_output, (_, _) = self.lstm2(lstm_output)
            assert lstm_output.size() == (y_len, bs, self.hidden_dim)

        # word scores
        output = F.dropout(lstm_output, p=self.dropout, training=self.training).view(-1, self.hidden_dim)
        if self.lstm_proj_layer is not None:
            output = F.relu(self.lstm_proj_layer(output))
        scores = self.proj(output).view(y_len, bs, self.n_words)
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
            - LongTensor of size (batch_size,), sentence x_len
        """
        latent = encoded.dec_input
        x_len = encoded.input_len
        is_cuda = latent.is_cuda
        one_hot = None  # [] if temperature is not None else None

        # check inputs
        assert latent.size() == (x_len.max(), x_len.size(0), self.emb_dim)
        assert attr.size() == (x_len.size(0), len(self.attributes))
        assert (sample is True) ^ (temperature is None)

        # initialize generated sentences batch
        slen, bs = latent.size(0), latent.size(1)
        assert x_len.max() == slen and x_len.size(0) == bs
        cur_len = 1
        decoded = torch.LongTensor(max_len, bs).fill_(self.pad_index)
        decoded = decoded.cuda() if is_cuda else decoded
        decoded[0] = self.bos_index

        # compute attention
        mask = get_mask(x_len, True, cuda=is_cuda) == 0
        h_c_1, h_c_2 = None, None
        hidden_states = [latent.data.new(1, bs, self.hidden_dim).zero_()]

        while cur_len < max_len:
            # previous word embeddings
            if cur_len == 1 and self.bos_attr != '':
                embeddings = self.get_bos_attr(attr)
            else:
                embeddings = self.embeddings(decoded[cur_len - 1])
            embeddings = F.dropout(embeddings, p=self.dropout, training=self.training)

            # attention layer
            attention = self.get_attention(latent, hidden_states[-1][0], embeddings, mask)

            # lstm step
            lstm_input = embeddings.unsqueeze(0)
            if self.input_feeding:
                lstm_input = torch.cat([lstm_input, attention], 2)
            lstm_output, h_c_1 = self.lstm1(lstm_input, h_c_1)
            assert lstm_output.size() == (1, bs, self.hidden_dim)
            hidden_states.append(lstm_output)

            # lstm (layers > 1)
            if self.n_dec_layers > 1:
                lstm_output = F.dropout(lstm_output, p=self.dropout, training=self.training)
                if not self.input_feeding:
                    lstm_output = torch.cat([lstm_output, attention], 2)
                lstm_output, h_c_2 = self.lstm2(lstm_output, h_c_2)
                assert lstm_output.size() == (1, bs, self.hidden_dim)

            # word scores
            output = F.dropout(lstm_output, p=self.dropout, training=self.training).view(-1, self.hidden_dim)
            if self.lstm_proj_layer is not None:
                output = F.relu(self.lstm_proj_layer(output))
            scores = self.proj(output).view(bs, self.n_words)
            if self.bias_attr != '':
                scores = scores + self.get_bias_attr(attr)
            scores = scores.data

            # select next words: sample (Gumbel Softmax) or one-hot
            if sample:
                # if temperature is not None:
                #     gumbel = gumbel_softmax(scores, temperature, hard=True)
                #     next_words = gumbel.max(1)[1]
                #     one_hot.append(gumbel)
                # else:
                next_words = torch.multinomial(F.softmax(scores / temperature, dim=1), 1).squeeze(1)
            else:
                next_words = torch.topk(scores, 1)[1].squeeze(1)
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


def build_lstm_enc_dec(params):
    logger.info("============ Building LSTM attention model - Encoder ...")
    encoder = Encoder(params)
    logger.info("")
    logger.info("============ Building LSTM attention model - Decoder ...")
    decoder = Decoder(params, encoder)
    logger.info("")
    return encoder, decoder


def build_transformer_enc_dec(params):
    from .transformer import TransformerEncoder, TransformerDecoder

    params.left_pad_source = False
    params.left_pad_target = False

    assert hasattr(params, 'dropout')
    assert hasattr(params, 'attention_dropout')
    assert hasattr(params, 'relu_dropout')

    params.encoder_embed_dim = params.emb_dim
    params.encoder_ffn_embed_dim = params.transformer_ffn_emb_dim
    params.encoder_layers = params.n_enc_layers
    assert hasattr(params, 'encoder_attention_heads')
    assert hasattr(params, 'encoder_normalize_before')

    params.decoder_embed_dim = params.emb_dim
    params.decoder_ffn_embed_dim = params.transformer_ffn_emb_dim
    params.decoder_layers = params.n_dec_layers
    assert hasattr(params, 'decoder_attention_heads')
    assert hasattr(params, 'decoder_normalize_before')

    logger.info("============ Building transformer attention model - Encoder ...")
    encoder = TransformerEncoder(params)
    logger.info("")
    logger.info("============ Building transformer attention model - Decoder ...")
    decoder = TransformerDecoder(params, encoder)
    logger.info("")
    return encoder, decoder


def build_attention_model(params, data, cuda=True):
    """
    Build a encoder / decoder, and the decoder reconstruction loss function.
    """
    # encoder / decoder / discriminator
    if params.transformer:
        encoder, decoder = build_transformer_enc_dec(params)
    else:
        encoder, decoder = build_lstm_enc_dec(params)
    if params.lambda_dis != "0":
        logger.info("============ Building attention model - Discriminator ...")
        if params.disc_lstm_dim > 0:
            assert params.disc_lstm_layers >= 1
            discriminator = MultiAttrDiscriminatorLSTM(params)
        else:
            discriminator = MultiAttrDiscriminator(params)
        logger.info("")
    else:
        discriminator = None

    # loss function for decoder reconstruction
    loss_weight = torch.FloatTensor(params.n_words).fill_(1)
    loss_weight[params.pad_index] = 0
    if params.label_smoothing <= 0:
        decoder.loss_fn = nn.CrossEntropyLoss(loss_weight, size_average=True)
    else:
        decoder.loss_fn = LabelSmoothedCrossEntropyLoss(
            params.label_smoothing,
            params.pad_index,
            size_average=True,
            weight=loss_weight,
        )

    # language model
    if params.lambda_lm != "0":
        logger.info("============ Building attention model - Language model ...")
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
                enc = reloaded.get('enc', reloaded.get('encoder'))
                reload_model(encoder, enc, encoder.ENC_ATTR)
            if params.reload_dec:
                logger.info("Reloading decoder...")
                dec = reloaded.get('dec', reloaded.get('decoder'))
                reload_model(decoder, dec, decoder.DEC_ATTR)
            if params.reload_dis:
                assert discriminator is not None
                logger.info("Reloading discriminator...")
                dis = reloaded.get('dis', reloaded.get('discriminator'))
                reload_model(discriminator, dis, discriminator.DIS_ATTR)

    # log models
    encdec_params = set(
        p
        for module in [encoder, decoder]
        for p in module.parameters()
        if p.requires_grad
    )
    num_encdec_params = sum(p.numel() for p in encdec_params)
    logger.info("============ Model summary")
    logger.info("Number of enc+dec parameters: {}".format(num_encdec_params))
    logger.info("Encoder: {}".format(encoder))
    logger.info("Decoder: {}".format(decoder))
    logger.info("Discriminator: {}".format(discriminator))
    logger.info("LM: {}".format(lm))
    logger.info("")

    return encoder, decoder, discriminator, lm
