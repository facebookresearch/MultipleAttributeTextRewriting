# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import time
from logging import getLogger
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

# from .utils import clip_parameters
from .utils import get_optimizer, parse_lambda_config, update_lambdas
from .attributes import gen_rand_attr
from .model import build_ae_model
from .multiprocessing_event_loop import MultiprocessingEventLoop
from .test import test_sharing


logger = getLogger()


class TrainerBase(MultiprocessingEventLoop):

    def __init__(self, data, params):
        """
        Initialize trainer.
        """
        if hasattr(params, 'otf_num_processes'):
            super().__init__(device_ids=tuple(range(params.otf_num_processes)))

        # epoch size
        if params.epoch_size == -1:
            params.epoch_size = len(data['mono']['train'])
        assert params.epoch_size > 0

        # stopping criterion
        if params.stop_crit == '':
            self.crit = None
            self.crit_best = None
        else:
            crit, dec_max = params.stop_crit.split(',')
            assert crit[0] in ['_', '+'] and dec_max.isdigit()
            sign = 1 if crit[0] == '+' else -1
            self.crit = crit[1:]
            self.crit_best = sign * -1e12
            self.crit_sign = sign
            self.decrease = 0
            self.decrease_max = int(dec_max)

        # validation metrics
        self.metrics_best = {}
        self.metrics_sign = {}
        metrics = [m for m in params.metrics.split(',') if m != '']
        for metric in metrics:
            assert metric[0] in ['_', '+']
            sign = 1 if metric[0] == '+' else -1
            self.metrics_best[metric[1:]] = sign * -1e12
            self.metrics_sign[metric[1:]] = sign

        # periodic save with optional conditions
        if params.save_periodic == '':
            self.save_periodic_config = False
        else:
            split = params.save_periodic.split(',')
            assert split[0].isdigit()
            period = int(split[0])
            conditions = [x.split(':') for x in split[1:]]
            assert period >= 1
            assert all([len(x) == 2 and len(x[0]) >= 1 and len(x[1]) >= 2 and x[1][0] in ['+', '-'] for x in conditions])
            conditions = [(name, 1 if sign_value[0] == '+' else -1, float(sign_value[1:])) for name, sign_value in conditions]
            self.save_periodic_config = (period, conditions)

    def get_iter_name(self, iter_name, attr_label):
        """
        Create an iterator name.
        """
        if attr_label is None:
            return iter_name
        i, j = attr_label
        attr = self.params.attributes[i]
        label = self.params.id2label[attr][j]
        return ','.join([x for x in [iter_name, attr, label] if x is not None])

    def get_iterator(self, iter_name, attr_label):
        """
        Create a new iterator for a dataset.
        """
        key = self.get_iter_name(iter_name, attr_label)
        logger.info("Creating new training %s iterator ..." % key)
        dataset = self.data['mono']['train']
        iterator = dataset.get_iterator(
            shuffle=True, group_by_size=self.params.group_by_size,
            n_sentences=-1, attr_label=attr_label
        )()
        self.iterators[key] = iterator
        return iterator

    def get_batch(self, iter_name, attr_label=None):
        """
        Return a batch of sentences from a dataset.
        """
        key = self.get_iter_name(iter_name, attr_label)
        iterator = self.iterators.get(key, None)
        if iterator is None:
            iterator = self.get_iterator(iter_name, attr_label)
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = self.get_iterator(iter_name, attr_label)
            batch = next(iterator)
        return batch

    def get_attr_label(self):
        """
        Generate an attribute / label for balanced training.
        """
        if self.params.balanced_train:
            attr_id = np.random.randint(len(self.params.attr_values))
            label_id = np.random.randint(len(self.params.attr_values[self.params.attributes[attr_id]]))
            return (attr_id, label_id)
        else:
            return None

    def word_shuffle(self, x, l):
        """
        Randomly shuffle input words.
        """
        if self.params.word_shuffle == 0:
            return x, l

        # define noise word scores
        noise = np.random.uniform(0, self.params.word_shuffle, size=(x.size(0) - 1, x.size(1)))
        noise[0] = -1  # do not move start sentence symbol

        # be sure to shuffle entire words
        bpe_end = self.bpe_end[x]
        word_idx = bpe_end[::-1].cumsum(0)[::-1]
        word_idx = word_idx.max(0)[None, :] - word_idx

        assert self.params.word_shuffle > 1
        x2 = x.clone()
        for i in range(l.size(0)):
            # generate a random permutation
            scores = word_idx[:l[i] - 1, i] + noise[word_idx[:l[i] - 1, i], i]
            scores += 1e-6 * np.arange(l[i] - 1)  # ensure no reordering inside a word
            permutation = scores.argsort()
            # shuffle words
            x2[:l[i] - 1, i].copy_(x2[:l[i] - 1, i][torch.from_numpy(permutation)])
        return x2, l

    def word_dropout(self, x, l):
        """
        Randomly drop input words.
        """
        if self.params.word_dropout == 0:
            return x, l
        assert 0 < self.params.word_dropout < 1

        # define words to drop
        bos_index = self.params.bos_index
        assert (x[0] == bos_index).sum() == l.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.params.word_dropout
        keep[0] = 1  # do not drop the start sentence symbol

        # be sure to drop entire words
        bpe_end = self.bpe_end[x]
        word_idx = bpe_end[::-1].cumsum(0)[::-1]
        word_idx = word_idx.max(0)[None, :] - word_idx

        sentences = []
        lengths = []
        for i in range(l.size(0)):
            assert x[l[i] - 1, i] == self.params.eos_index
            words = x[:l[i] - 1, i].tolist()
            # randomly drop words from the input
            new_s = [w for j, w in enumerate(words) if keep[word_idx[j, i], i]]
            # we need to have at least one word in the sentence (more than the start / end sentence symbols)
            if len(new_s) == 1:
                new_s.append(words[np.random.randint(1, len(words))])
            new_s.append(self.params.eos_index)
            assert len(new_s) >= 3 and new_s[0] == bos_index and new_s[-1] == self.params.eos_index
            sentences.append(new_s)
            lengths.append(len(new_s))
        # re-construct input
        l2 = torch.LongTensor(lengths)
        x2 = torch.LongTensor(l2.max().item(), l2.size(0)).fill_(self.params.pad_index)
        for i in range(l2.size(0)):
            x2[:l2[i], i].copy_(torch.LongTensor(sentences[i]))
        return x2, l2

    def word_blank(self, x, l):
        """
        Randomly blank input words.
        """
        if self.params.word_blank == 0:
            return x, l
        assert 0 < self.params.word_blank < 1

        # define words to blank
        bos_index = self.params.bos_index
        assert (x[0] == bos_index).sum() == l.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.params.word_blank
        keep[0] = 1  # do not blank the start sentence symbol

        # be sure to blank entire words
        bpe_end = self.bpe_end[x]
        word_idx = bpe_end[::-1].cumsum(0)[::-1]
        word_idx = word_idx.max(0)[None, :] - word_idx

        sentences = []
        for i in range(l.size(0)):
            assert x[l[i] - 1, i] == self.params.eos_index
            words = x[:l[i] - 1, i].tolist()
            # randomly blank words from the input
            new_s = [w if keep[word_idx[j, i], i] else self.params.blank_index for j, w in enumerate(words)]
            new_s.append(self.params.eos_index)
            assert len(new_s) == l[i] and new_s[0] == bos_index and new_s[-1] == self.params.eos_index
            sentences.append(new_s)
        # re-construct input
        x2 = torch.LongTensor(l.max().item(), l.size(0)).fill_(self.params.pad_index)
        for i in range(l.size(0)):
            x2[:l[i], i].copy_(torch.LongTensor(sentences[i]))
        return x2, l

    def add_noise(self, words, lengths):
        """
        Add noise to the encoder input.
        """
        words, lengths = self.word_shuffle(words, lengths)
        words, lengths = self.word_dropout(words, lengths)
        words, lengths = self.word_blank(words, lengths)
        return words, lengths

    def zero_grad(self, models):
        """
        Zero gradients.
        """
        if type(models) is not list:
            models = [models]
        models = [self.model_opt[name] for name in models]
        for _, optimizer in models:
            if optimizer is not None:
                optimizer.zero_grad()

    def update_params(self, models):
        """
        Update parameters.
        """
        if type(models) is not list:
            models = [models]
        # don't update encoder when it's frozen
        models = [self.model_opt[name] for name in models]
        # clip gradients
        for model, _ in models:
            clip_grad_norm_(model.parameters(), self.params.clip_grad_norm)

        # optimizer
        for _, optimizer in models:
            if optimizer is not None:
                optimizer.step()

    def get_lrs(self, models):
        """
        Get current optimizer learning rates.
        """
        if type(models) is not list:
            models = [models]
        lrs = {}
        for name in models:
            optimizer = self.model_opt[name][1]
            if optimizer is not None:
                lrs[name] = optimizer.param_groups[0]['lr']
        return lrs

    def save_best_model(self, scores):
        """
        Save best models according to given validation metrics.
        """
        for metric in self.metrics_best.keys():
            sign = self.metrics_sign[metric]
            if scores[metric] * sign > self.metrics_best[metric] * sign:
                self.metrics_best[metric] = scores[metric]
                logger.info('New best score for %s: %.6f' % (metric, scores[metric]))
                self.save_model('best-%s' % metric)

    def save_periodic(self, scores):
        """
        Save the models periodically.
        """
        save_periodic = self.save_periodic_config
        if save_periodic is False:
            return
        period, conditions = save_periodic
        if (period > 0 and self.epoch > 0 and self.epoch % period == 0 and
                all([sign_factor * scores[name] >= sign_factor * value
                     for name, sign_factor, value in conditions])):
            self.save_model('periodic-%i' % self.epoch)

    def stop_experiment(self, scores):
        """
        End the experiment if the stopping criterion is reached.
        """
        if self.crit is None:
            return
        assert self.crit in scores
        sign = self.crit_sign
        # update best criterion / decrease counts
        if scores[self.crit] * sign > self.crit_best * sign:
            self.decrease = 0
            self.crit_best = scores[self.crit]
            logger.info("New best validation score: %f" % self.crit_best)
        else:
            self.decrease += 1
            logger.info("Not a better validation score (%i / %i)."
                        % (self.decrease, self.decrease_max))
        # stop the experiment if there is no more improvement
        if self.decrease > self.decrease_max:
            logger.info("Stopping criterion has been below its best value for more "
                        "than %i epochs. Ending the experiment..." % self.decrease_max)
            exit()

    def end_epoch(self, scores):
        """
        End the epoch.
        """
        self.save_best_model(scores)
        self.save_periodic(scores)
        self.stop_experiment(scores)
        self.save_checkpoint()
        self.epoch += 1

    def iter(self):
        raise NotImplementedError

    def print_stats(self):
        raise NotImplementedError

    def save_model(self, name):
        raise NotImplementedError

    def save_checkpoint(self):
        raise NotImplementedError

    def reload_checkpoint(self):
        raise NotImplementedError


class TrainerAE(TrainerBase):

    def __init__(self, encoder, decoder, discriminator, lm, data, params):
        """
        Initialize trainer.
        """
        super().__init__(data, params)
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.lm = lm
        self.data = data
        self.params = params

        # initialization for on-the-fly generation/training
        if params.train_bt:
            self.otf_start_multiprocessing()

        # define encoder parameters (the ones shared with the
        # decoder are optimized by the decoder optimizer)
        enc_params = list(encoder.parameters())
        assert enc_params[0].size() == (params.n_words, params.emb_dim)
        if self.params.share_encdec_emb:
            enc_params = enc_params[1:]

        # optimizers
        if params.dec_optimizer == 'enc_optimizer':
            params.dec_optimizer = params.enc_optimizer
        self.enc_optimizer = get_optimizer(enc_params, params.enc_optimizer) if len(enc_params) > 0 else None
        self.dec_optimizer = get_optimizer(decoder.parameters(), params.dec_optimizer)
        self.dis_optimizer = get_optimizer(discriminator.parameters(), params.dis_optimizer) if discriminator is not None else None
        self.lm_optimizer = get_optimizer(lm.parameters(), params.enc_optimizer) if lm is not None else None

        # models / optimizers
        self.model_opt = {
            'enc': (self.encoder, self.enc_optimizer),
            'dec': (self.decoder, self.dec_optimizer),
            'dis': (self.discriminator, self.dis_optimizer),
            'lm': (self.lm, self.lm_optimizer),
        }

        # training variables
        self.epoch = 0
        self.n_total_iter = 0
        self.freeze_enc_emb = self.params.freeze_enc_emb
        self.freeze_dec_emb = self.params.freeze_dec_emb

        # training statistics
        self.n_iter = 0
        self.n_sentences = 0
        self.stats = {
            'processed_s': 0,
            'processed_w': 0,
        }
        self.stats['xe_ae'] = []
        self.stats['xe_bt'] = []
        self.stats['xe_lm'] = []
        self.stats['dis_costs'] = []
        self.last_time = time.time()
        if params.train_bt:
            self.gen_time = 0

        # data iterators
        self.iterators = {}

        # initialize BPE subwords
        self.init_bpe()

        # initialize lambda coefficients / sampling temperature,  and their configurations
        parse_lambda_config(params, 'lambda_ae')
        parse_lambda_config(params, 'lambda_bt')
        parse_lambda_config(params, 'lambda_lm')
        parse_lambda_config(params, 'lambda_dis')
        parse_lambda_config(params, 'otf_temperature')

    def init_bpe(self):
        """
        Index BPE words.
        """
        dico = self.data['dico']
        self.bpe_end = np.array([not dico[i].endswith('@@') for i in range(len(dico))])

    # def discriminator_step(self):
    #     """
    #     Train the discriminator on the latent space.
    #     """
    #     self.encoder.eval()
    #     self.decoder.eval()
    #     self.discriminator.train()

    #     # train on monolingual data only
    #     if self.params.n_mono == 0:
    #         raise Exception("No data to train the discriminator!")

    #     # batch / encode
    #     encoded = []
    #     for lang_id, lang in enumerate(self.params.langs):
    #         sent1, len1 = self.get_batch('dis')
    #         with torch.no_grad():
    #             encoded.append(self.encoder(sent1.cuda(), len1, lang_id))

    #     # discriminator
    #     dis_inputs = [x.dis_input.view(-1, x.dis_input.size(-1)) for x in encoded]
    #     ntokens = [dis_input.size(0) for dis_input in dis_inputs]
    #     encoded = torch.cat(dis_inputs, 0)
    #     predictions = self.discriminator(encoded.data)

    #     # loss
    #     self.dis_target = torch.cat([torch.zeros(sz).fill_(i) for i, sz in enumerate(ntokens)])
    #     self.dis_target = self.dis_target.contiguous().long().cuda()
    #     y = self.dis_target

    #     loss = F.cross_entropy(predictions, y)
    #     self.stats['dis_costs'].append(loss.item())

    #     # optimizer
    #     self.zero_grad('dis')
    #     loss.backward()
    #     self.update_params('dis')
    #     clip_parameters(self.discriminator, self.params.dis_clip)

    # def lm_step(self):
    #     """
    #     Language model training.
    #     """
    #     self.lm.train()

    #     # batch
    #     sent, lengths, attr = self.get_batch('lm')
    #     sent, attr = sent.cuda(), attr.cuda()

    #     # forward
    #     scores = self.lm(sent[:-1], lengths - 1, attr, True, False)

    #     # loss
    #     loss = F.cross_entropy(scores.view(-1, self.params.n_words), sent[1:].view(-1))
    #     self.stats['xe_lm'].append(loss.item())
    #     loss = self.params.lambda_lm * loss

    #     # check NaN
    #     if (loss != loss).data.any():
    #         logger.error("NaN detected")
    #         exit()

    #     # optimizer
    #     self.zero_grad(['lm'])
    #     loss.backward()
    #     self.update_params(['lm'])

    #     # number of processed sentences / words
    #     self.stats['processed_s'] += lengths.size(0)
    #     self.stats['processed_w'] += lengths.sum()

    def enc_dec_step(self, lambda_xe, attr_label):
        """
        Source / target autoencoder training (parallel data):
            - encoders / decoders training on cross-entropy
            - encoders training on discriminator feedback
            - encoders training on L2 loss (seq2seq only, not for attention)
        """
        params = self.params
        n_words = params.n_words
        self.encoder.train()
        self.decoder.train()
        if self.discriminator is not None:
            self.discriminator.eval()

        # batch
        sent1, len1, attr1 = self.get_batch('encdec', attr_label)
        sent2, len2, attr2 = sent1, len1, attr1

        # prepare the encoder / decoder inputs
        sent1, len1 = self.add_noise(sent1, len1)
        sent1, sent2 = sent1.cuda(), sent2.cuda()
        attr1, attr2 = attr1.cuda(), attr2.cuda()

        # encoded states
        encoded = self.encoder(sent1, len1)

        # cross-entropy scores / loss
        scores = self.decoder(encoded, sent2[:-1], attr2)
        xe_loss = self.decoder.loss_fn(scores.view(-1, n_words), sent2[1:].view(-1))
        self.stats['xe_ae'].append(xe_loss.item())

        # discriminator feedback loss
        if params.lambda_dis:
            if params.disc_lstm_dim > 0:
                predictions = self.discriminator(encoded.dis_input, encoded.input_len)
            else:
                predictions = self.discriminator(
                    encoded.dis_input.view(-1, encoded.dis_input.size(-1))
                )

            fake_y = torch.LongTensor(predictions.size(0)).random_(1, params.n_langs)
            fake_y = (fake_y + lang1_id) % params.n_langs
            fake_y = fake_y.cuda()
            dis_loss = F.cross_entropy(predictions, fake_y)

        # total loss
        assert lambda_xe > 0
        loss = lambda_xe * xe_loss
        if params.lambda_dis:
            loss = loss + params.lambda_dis * dis_loss

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        # optimizer
        self.zero_grad(['enc', 'dec'])
        loss.backward()
        self.update_params(['enc', 'dec'])

        # number of processed sentences / words
        self.stats['processed_s'] += len2.size(0)
        self.stats['processed_w'] += len2.sum()

    def otf_start_multiprocessing(self):
        logger.info("Starting subprocesses for OTF generation ...")

        # initialize subprocesses
        for rank in range(self.num_replicas):
            self.call_async(rank, '_async_otf_init', params=self.params)

    def _async_otf_init(self, rank, device_id, params):
        # build model on subprocess

        from copy import deepcopy
        params = deepcopy(params)
        self.params = params
        self.params.cpu_thread = True
        self.data = None  # do not load data in the CPU threads
        self.iterators = {}
        self.encoder, self.decoder, _, _ = build_ae_model(self.params, self.data, cuda=False)

    def otf_sync_params(self):
        """
        Synchronize CPU threads parameters.
        """
        def get_flat_params(module):
            return torch._utils._flatten_dense_tensors(
                [p.data for p in module.parameters()])

        encoder_params = get_flat_params(self.encoder).cpu().share_memory_()
        decoder_params = get_flat_params(self.decoder).cpu().share_memory_()

        for rank in range(self.num_replicas):
            self.call_async(
                rank,
                '_async_otf_sync_params',
                encoder_params=encoder_params,
                decoder_params=decoder_params
            )

    def _async_otf_sync_params(self, rank, device_id, encoder_params, decoder_params):
        """
        Synchronize CPU threads parameters.
        """
        def set_flat_params(module, flat):
            params = [p.data for p in module.parameters()]
            for p, f in zip(params, torch._utils._unflatten_dense_tensors(flat, params)):
                p.copy_(f)

        # copy parameters back into modules
        set_flat_params(self.encoder, encoder_params)
        set_flat_params(self.decoder, decoder_params)

    def otf_sync_temperature(self):
        """
        Synchronize CPU threads sampling generation temperature.
        """
        for rank in range(self.num_replicas):
            self.call_async(rank, '_async_otf_sync_temperature', otf_temperature=self.params.otf_temperature)

    def _async_otf_sync_temperature(self, rank, device_id, otf_temperature):
        """
        Synchronize CPU threads sampling generation temperature.
        """
        self.params.otf_temperature = otf_temperature

    def otf_bt_gen_async(self, init_cache_size=None):
        logger.info("Populating initial OTF generation cache ...")
        if init_cache_size is None:
            init_cache_size = self.num_replicas
        # self._async_otf_bt_gen(0, 0, self.get_worker_batches())  # DEBUG ON GPU
        cache = [
            self.call_async(
                rank=i % self.num_replicas,
                action='_async_otf_bt_gen',
                result_type='otf_gen',
                fetch_all=True,
                batches=self.get_worker_batches()
            ) for i in range(init_cache_size)
        ]
        while True:
            results = cache[0].gen()
            for rank, _ in results:
                cache.pop(0)  # keep the cache a fixed size
                cache.append(
                    self.call_async(
                        rank=rank,
                        action='_async_otf_bt_gen',
                        result_type='otf_gen',
                        fetch_all=True,
                        batches=self.get_worker_batches()
                    )
                )
            for _, result in results:
                yield result

    def get_worker_batches(self):
        """
        Create batches for CPU threads.
        """
        assert self.params.lambda_bt > 0
        sent1, len1, attr1 = self.get_batch('otf', attr_label=self.get_attr_label())
        sent3, len3, attr3 = sent1, len1, attr1
        batches = [{
            'sent1': sent1,
            'sent3': sent3,
            'len1': len1,
            'len3': len3,
            'attr1': attr1,
            'attr3': attr3,
        }]
        return batches

    def _async_otf_bt_gen(self, rank, device_id, batches):
        """
        On the fly back-translation (generation step).
        """
        params = self.params
        self.encoder.eval()
        self.decoder.eval()

        results = []

        with torch.no_grad():

            for batch in batches:

                sent1, len1, attr1 = batch['sent1'], batch['len1'], batch['attr1']
                sent3, len3, attr3 = batch['sent3'], batch['len3'], batch['attr3']

                # generate random attributes
                attr2 = gen_rand_attr(len1.size(0), params.attributes, params.attr_values)

                # attr1 -> attr2
                encoded = self.encoder(sent1, len1)
                max_len = int(1.5 * len1.max() + 10)
                assert params.otf_temperature >= 0
                if params.otf_temperature == 0:
                    sent2, len2, _ = self.decoder.generate(encoded, attr2, max_len=max_len)
                else:
                    sent2, len2, _ = self.decoder.generate(encoded, attr2, max_len=max_len,
                                                           sample=True, temperature=params.otf_temperature)

                # keep cached batches on CPU for easier transfer
                assert not any(x.is_cuda for x in [sent1, sent2, sent3])
                assert not any(x.is_cuda for x in [attr1, attr2, attr3])
                results.append(dict([
                    ('sent1', sent1), ('len1', len1), ('attr1', attr1),
                    ('sent2', sent2), ('len2', len2), ('attr2', attr2),
                    ('sent3', sent3), ('len3', len3), ('attr3', attr3),
                ]))

        return (rank, results)

    def otf_bt(self, batch, lambda_xe, backprop_temperature):
        """
        On the fly back-translation.
        """
        params = self.params
        sent1, len1, attr1 = batch['sent1'], batch['len1'], batch['attr1']
        sent2, len2, attr2 = batch['sent2'], batch['len2'], batch['attr2']
        sent3, len3, attr3 = batch['sent3'], batch['len3'], batch['attr3']

        if lambda_xe == 0:
            logger.warning("Unused generated CPU batch!")
            return

        n_words2 = params.n_words
        n_words3 = params.n_words
        self.encoder.train()
        self.decoder.train()

        # prepare batch
        sent1, sent2, sent3 = sent1.cuda(), sent2.cuda(), sent3.cuda()
        attr1, attr2, attr3 = attr1.cuda(), attr2.cuda(), attr3.cuda()
        bs = sent1.size(1)

        if backprop_temperature == -1:
            # attr2 -> attr3
            encoded = self.encoder(sent2, len2)
        else:
            raise Exception("Not implemented for attributes yet! Need to add attribute embedding below.")
            # attr1 -> attr2
            encoded = self.encoder(sent1, len1)
            scores = self.decoder(encoded, sent2[:-1], attr2)
            assert scores.size() == (len2.max() - 1, bs, n_words2)

            # attr2 -> attr3
            bos = torch.cuda.FloatTensor(1, bs, n_words2).zero_()
            bos[0, :, params.bos_index] = 1
            sent2_input = torch.cat([bos, F.softmax(scores / backprop_temperature, -1)], 0)
            encoded = self.encoder(sent2_input, len2)

        # cross-entropy scores / loss
        scores = self.decoder(encoded, sent3[:-1], attr3)
        xe_loss = self.decoder.loss_fn(scores.view(-1, n_words3), sent3[1:].view(-1))
        self.stats['xe_bt'].append(xe_loss.item())
        assert lambda_xe > 0
        loss = lambda_xe * xe_loss

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        # optimizer
        assert params.otf_update_enc or params.otf_update_dec
        to_update = []
        if params.otf_update_enc:
            to_update.append('enc')
        if params.otf_update_dec:
            to_update.append('dec')
        self.zero_grad(to_update)
        loss.backward()
        self.update_params(to_update)

        # number of processed sentences / words
        self.stats['processed_s'] += len3.size(0)
        self.stats['processed_w'] += len3.sum()

    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        n_batches = 0
        n_batches += int(self.params.lambda_ae > 0)
        n_batches += int(self.params.lambda_bt > 0)
        self.n_sentences += n_batches * self.params.batch_size
        self.print_stats()
        update_lambdas(self.params, self.n_total_iter)

    def print_stats(self):
        """
        Print statistics about the training.
        """
        # average loss / statistics
        if self.n_iter % 50 == 0:

            mean_loss = []
            mean_loss.append(('XE-AE', 'xe_ae'))
            mean_loss.append(('XE-BT', 'xe_bt'))
            mean_loss.append(('XE-LM', 'xe_lm'))
            mean_loss.append(('DIS', 'dis_costs'))

            s_iter = "%7i - " % self.n_iter
            s_stat = ' || '.join(['{}: {:7.4f}'.format(k, np.mean(self.stats[l]))
                                 for k, l in mean_loss if len(self.stats[l]) > 0])
            for _, l in mean_loss:
                del self.stats[l][:]

            # processing speed
            new_time = time.time()
            diff = new_time - self.last_time
            s_speed = "{:7.2f} sent/s - {:8.2f} words/s - ".format(
                self.stats['processed_s'] * 1.0 / diff,
                self.stats['processed_w'] * 1.0 / diff
            )
            self.stats['processed_s'] = 0
            self.stats['processed_w'] = 0
            self.last_time = new_time

            lrs = self.get_lrs(['enc', 'dec'])
            s_lr = " - LR " + ",".join("{}={:.4e}".format(k, lr) for k, lr in lrs.items())

            # generation time
            if self.params.train_bt:
                s_time = " - Sentences generation time: % .2fs (%.2f%%)" % (self.gen_time, 100. * self.gen_time / diff)
                self.gen_time = 0
            else:
                s_time = ""

            # log speed + stats
            logger.info(s_iter + s_speed + s_stat + s_lr + s_time)

    def save_model(self, name):
        """
        Save the model.
        """
        path = os.path.join(self.params.dump_path, '%s.pth' % name)
        logger.info('Saving model to %s ...' % path)
        torch.save({
            'enc': self.encoder,
            'dec': self.decoder,
            'dis': self.discriminator,
            'lm': self.lm,
        }, path)

    def save_checkpoint(self):
        """
        Checkpoint the experiment.
        """
        data = {
            'encoder': self.encoder,
            'decoder': self.decoder,
            'discriminator': self.discriminator,
            'lm': self.lm,
            'enc_optimizer': self.enc_optimizer,
            'dec_optimizer': self.dec_optimizer,
            'dis_optimizer': self.dis_optimizer,
            'lm_optimizer': self.lm_optimizer,
            'epoch': self.epoch,
            'n_total_iter': self.n_total_iter,
            'metrics_best': self.metrics_best,
            'crit_best': self.crit_best,
        }
        path = os.path.join(self.params.dump_path, 'checkpoint.pth')
        logger.info("Saving checkpoint to %s ..." % path)
        torch.save(data, path)

    def reload_checkpoint(self):
        """
        Reload a checkpoint if we find one.
        """
        # reload checkpoint
        path = os.path.join(self.params.dump_path, 'checkpoint.pth')
        if not os.path.isfile(path):
            return
        logger.warning('Reloading checkpoint from %s ...' % path)
        data = torch.load(path)
        self.encoder = data['encoder']
        self.decoder = data['decoder']
        self.discriminator = data['discriminator']
        self.lm = data['lm']
        self.enc_optimizer = data['enc_optimizer']
        self.dec_optimizer = data['dec_optimizer']
        self.dis_optimizer = data['dis_optimizer']
        self.lm_optimizer = data['lm_optimizer']
        self.epoch = data['epoch'] + 1
        self.n_total_iter = data['n_total_iter']
        self.metrics_best = data['metrics_best']
        self.crit_best = data['crit_best']
        self.model_opt = {
            'enc': (self.encoder, self.enc_optimizer),
            'dec': (self.decoder, self.dec_optimizer),
            'dis': (self.discriminator, self.dis_optimizer),
            'lm': (self.lm, self.lm_optimizer),
        }
        logger.warning('Checkpoint reloaded. Resuming at epoch %i ...' % self.epoch)

    def test_sharing(self):
        """
        Test to check that parameters are shared correctly.
        """
        test_sharing(self.encoder, self.decoder, self.params)
        logger.info("Test: Parameters are shared correctly.")


class TrainerClassifier(TrainerBase):

    def __init__(self, classifier, data, params):
        """
        Initialize trainer.
        """
        super().__init__(data, params)
        self.classifier = classifier
        self.data = data
        self.params = params

        # optimizers
        self.clf_optimizer = get_optimizer(classifier.parameters(), params.clf_optimizer)

        # models / optimizers
        self.model_opt = {'clf': (self.classifier, self.clf_optimizer)}

        # training variables
        self.epoch = 0
        self.n_total_iter = 0

        # training statistics
        self.n_iter = 0
        self.n_sentences = 0
        self.stats = {
            'processed_s': 0,
            'processed_w': 0,
        }
        self.stats['xe_clf'] = []
        self.last_time = time.time()

        # data iterators
        self.iterators = {}

    def clf_step(self, attr_label):
        """
        Language model training.
        """
        self.classifier.train()

        # batch
        sent, lengths, attr = self.get_batch('clf', attr_label)
        sent, attr = sent.cuda(), attr.cuda()

        # forward
        scores = self.classifier(sent, lengths)

        # loss
        loss = 0
        offset = 0
        for i, a in enumerate(self.params.attributes):
            n = len(self.params.attr_values[a])
            loss += F.cross_entropy(scores[:, offset:offset + n], attr[:, i] - offset)
            offset += n
        self.stats['xe_clf'].append(loss.item())

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        # optimizer
        self.zero_grad(['clf'])
        loss.backward()
        self.update_params(['clf'])

        # number of processed sentences / words
        self.stats['processed_s'] += lengths.size(0)
        self.stats['processed_w'] += lengths.sum()

    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        self.n_sentences += self.params.batch_size
        self.print_stats()

    def print_stats(self):
        """
        Print statistics about the training.
        """
        # average loss / statistics
        if self.n_iter % 200 == 0:
            mean_loss = [('XE-CLF', 'xe_clf')]

            s_iter = "%7i - " % self.n_iter
            s_stat = ' || '.join(['{}: {:7.4f}'.format(k, np.mean(self.stats[l]))
                                 for k, l in mean_loss if len(self.stats[l]) > 0])
            for _, l in mean_loss:
                del self.stats[l][:]

            # processing speed
            new_time = time.time()
            diff = new_time - self.last_time
            s_speed = "{:7.2f} sent/s - {:8.2f} words/s - ".format(
                self.stats['processed_s'] * 1.0 / diff,
                self.stats['processed_w'] * 1.0 / diff
            )
            self.stats['processed_s'] = 0
            self.stats['processed_w'] = 0
            self.last_time = new_time

            lrs = self.get_lrs(['clf'])
            s_lr = " - LR " + ",".join("{}={:.4e}".format(k, lr) for k, lr in lrs.items())

            # log speed + stats
            logger.info(s_iter + s_speed + s_stat + s_lr)

    def save_model(self, name):
        """
        Save the model.
        """
        path = os.path.join(self.params.dump_path, '%s.pth' % name)
        logger.info('Saving model to %s ...' % path)
        torch.save({'clf': self.classifier}, path)

    def save_checkpoint(self):
        """
        Checkpoint the experiment.
        """
        data = {
            'classifier': self.classifier,
            'clf_optimizer': self.clf_optimizer,
            'epoch': self.epoch,
            'n_total_iter': self.n_total_iter,
            'metrics_best': self.metrics_best,
            'crit_best': self.crit_best,
        }
        path = os.path.join(self.params.dump_path, 'checkpoint.pth')
        logger.info("Saving checkpoint to %s ..." % path)
        torch.save(data, path)

    def reload_checkpoint(self):
        """
        Reload a checkpoint if we find one.
        """
        # reload checkpoint
        path = os.path.join(self.params.dump_path, 'checkpoint.pth')
        if not os.path.isfile(path):
            return
        logger.warning('Reloading checkpoint from %s ...' % path)
        data = torch.load(path)
        self.classifier = data['classifier']
        self.clf_optimizer = data['clf_optimizer']
        self.epoch = data['epoch']
        self.n_total_iter = data['n_total_iter']
        self.metrics_best = data['metrics_best']
        self.crit_best = data['crit_best']
        self.model_opt = {'clf': (self.classifier, self.clf_optimizer)}
        logger.warning('Checkpoint reloaded. Resuming at epoch %i ...' % self.epoch)


class TrainerLM(TrainerBase):

    def __init__(self, lm, data, params):
        """
        Initialize trainer.
        """
        super().__init__(data, params)
        self.lm = lm
        self.data = data
        self.params = params

        # optimizers
        self.lm_optimizer = get_optimizer(lm.parameters(), params.lm_optimizer)

        # models / optimizers
        self.model_opt = {'lm': (self.lm, self.lm_optimizer)}

        # training variables
        self.epoch = 0
        self.n_total_iter = 0

        # training statistics
        self.n_iter = 0
        self.n_sentences = 0
        self.stats = {
            'processed_s': 0,
            'processed_w': 0,
        }
        self.stats['xe_lm'] = []
        self.last_time = time.time()

        # data iterators
        self.iterators = {}

    def lm_step(self, attr_label):
        """
        Language model training.
        """
        self.lm.train()

        # batch
        sent, lengths, attr = self.get_batch('lm', attr_label)
        sent, attr = sent.cuda(), attr.cuda()

        # forward
        scores = self.lm(sent[:-1], lengths - 1, attr)

        # loss
        loss = F.cross_entropy(scores.view(-1, self.params.n_words), sent[1:].view(-1))
        self.stats['xe_lm'].append(loss.item())

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        # optimizer
        self.zero_grad(['lm'])
        loss.backward()
        self.update_params(['lm'])

        # number of processed sentences / words
        self.stats['processed_s'] += lengths.size(0)
        self.stats['processed_w'] += lengths.sum()

    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        self.n_sentences += self.params.batch_size
        self.print_stats()

    def print_stats(self):
        """
        Print statistics about the training.
        """
        # average loss / statistics
        if self.n_iter % 200 == 0:
            mean_loss = [('XE-LM', 'xe_lm')]

            s_iter = "%7i - " % self.n_iter
            s_stat = ' || '.join(['{}: {:7.4f}'.format(k, np.mean(self.stats[l]))
                                 for k, l in mean_loss if len(self.stats[l]) > 0])
            for _, l in mean_loss:
                del self.stats[l][:]

            # processing speed
            new_time = time.time()
            diff = new_time - self.last_time
            s_speed = "{:7.2f} sent/s - {:8.2f} words/s - ".format(
                self.stats['processed_s'] * 1.0 / diff,
                self.stats['processed_w'] * 1.0 / diff
            )
            self.stats['processed_s'] = 0
            self.stats['processed_w'] = 0
            self.last_time = new_time

            lrs = self.get_lrs(['lm'])
            s_lr = " - LR " + ",".join("{}={:.4e}".format(k, lr) for k, lr in lrs.items())

            # log speed + stats
            logger.info(s_iter + s_speed + s_stat + s_lr)

    def save_model(self, name):
        """
        Save the model.
        """
        path = os.path.join(self.params.dump_path, '%s.pth' % name)
        logger.info('Saving model to %s ...' % path)
        torch.save({'lm': self.lm}, path)

    def save_checkpoint(self):
        """
        Checkpoint the experiment.
        """
        data = {
            'lm': self.lm,
            'lm_optimizer': self.lm_optimizer,
            'epoch': self.epoch,
            'n_total_iter': self.n_total_iter,
            'metrics_best': self.metrics_best,
            'crit_best': self.crit_best,
        }
        path = os.path.join(self.params.dump_path, 'checkpoint.pth')
        logger.info("Saving checkpoint to %s ..." % path)
        torch.save(data, path)

    def reload_checkpoint(self):
        """
        Reload a checkpoint if we find one.
        """
        # reload checkpoint
        path = os.path.join(self.params.dump_path, 'checkpoint.pth')
        if not os.path.isfile(path):
            return
        logger.warning('Reloading checkpoint from %s ...' % path)
        data = torch.load(path)
        self.lm = data['lm']
        self.lm_optimizer = data['lm_optimizer']
        self.epoch = data['epoch']
        self.n_total_iter = data['n_total_iter']
        self.metrics_best = data['metrics_best']
        self.crit_best = data['crit_best']
        self.model_opt = {'lm': (self.lm, self.lm_optimizer)}
        logger.warning('Checkpoint reloaded. Resuming at epoch %i ...' % self.epoch)
