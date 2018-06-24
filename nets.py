'''
uni-LSTM + Virtual Adversarial Training
'''

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable

from six.moves import xrange

to_cpu = chainer.cuda.to_cpu
to_gpu = chainer.cuda.to_gpu

def kl_loss(xp, p_logit, q_logit):
    p = F.softmax(p_logit)
    _kl = F.sum(p * (F.log_softmax(p_logit) - F.log_softmax(q_logit)), 1)
    return F.sum(_kl) / xp.prod(xp.array(_kl.shape))


def get_normalized_vector(d, xp=None, shape=None):
    if shape is None:
        shape = tuple(range(1, len(d.shape)))
    d_norm = d
    if xp is not None:
        d_norm = d / (1e-12 + xp.max(xp.abs(d), shape, keepdims=True))
        d_norm = d_norm / xp.sqrt(1e-6 + xp.sum(d_norm ** 2, shape, keepdims=True))
    else:
        d_term = 1e-12 + F.max(F.absolute(d), shape, keepdims=True)
        d_norm = d / F.broadcast_to(d_term, d.shape)
        d_term = F.sqrt(1e-6 + F.sum(d ** 2, shape, keepdims=True))
        d_norm = d / F.broadcast_to(d_term, d.shape)
    return d_norm


class uniLSTM_iVAT(chainer.Chain):

    def __init__(self, n_vocab=None, emb_dim=256, hidden_dim=1024,
                 use_dropout=0.50, n_layers=1, hidden_classifier=30,
                 use_adv=0, xi_var=5.0, n_class=2,
                 args=None):
        self.args = args
        super(uniLSTM_iVAT, self).__init__(
            word_embed = L.EmbedID(n_vocab, emb_dim, ignore_label=-1),
            hidden_layer=L.Linear(hidden_dim, hidden_classifier),
            output_layer=L.Linear(hidden_classifier, n_class)
        )
        uni_lstm = L.NStepLSTM(n_layers=n_layers, in_size=emb_dim,
                               out_size=hidden_dim, dropout=use_dropout)
        # Forget gate bias => 1.0
        # MEMO: Values 1 and 5 reference the forget gate.
        for w in uni_lstm:
            w.b1.data[:] = 1.0
            w.b5.data[:] = 1.0

        self.add_link('uni_lstm', uni_lstm)

        self.hidden_dim = hidden_dim
        self.train = True
        self.use_dropout = use_dropout
        self.n_layers = n_layers
        self.use_adv = use_adv
        self.xi_var = xi_var
        self.n_vocab = n_vocab
        self.grad_scale = None

    def freeze_word_emb(self):
        self.word_embed.W.update_rule.enabled = False

    def set_pretrained_lstm(self, pretrain_model, word_only=True):
        # set word embeddding
        limit = self.word_embed.W.shape[0]
        self.word_embed.W.data[:] = pretrain_model.embed.W.data[:limit]

        if word_only:
            return True

        def split_weights(weights):
            input_dim = weights.shape[-1]
            reshape_weights = F.reshape(weights, (-1, 4, input_dim))
            reshape_weights = [reshape_weights[:, i, :] for i in xrange(4)]
            return reshape_weights

        def split_bias(bias):
            reshape_bias = F.reshape(bias, (-1, 4))
            reshape_bias = [reshape_bias[:, i] for i in xrange(4)]
            # reshape_bias = bias
            # reshape_bias = [reshape_bias[i::4] for i in xrange(4)]
            return reshape_bias

        # set lstm params
        pretrain_lstm = pretrain_model.lstm
        for layer_i in xrange(self.args.n_layers):
            w = pretrain_lstm[layer_i]
            source_w = [w.w2, w.w0, w.w1, w.w3, w.w6, w.w4, w.w5, w.w7]
            source_b = [w.b2, w.b0, w.b1, w.b3, w.b6, w.b4, w.b5, w.b7]

            w = self.uni_lstm[layer_i]
            # [NStepLSTM]
            # w0, w4 : input gate   (i)
            # w1, w5 : forget gate  (f)
            # w2, w6 : new memory gate (c)
            # w3, w7 : output gate

            # [Chaner LSTM]
            # a,   :   w2, w6
            # i,   :   w0, w4
            # f,   :   w1, w5
            # o    :   w3, w7
            uni_lstm_w = [w.w2, w.w0, w.w1, w.w3, w.w6, w.w4, w.w5, w.w7]
            uni_lstm_b = [w.b2, w.b0, w.b1, w.b3, w.b6, w.b4, w.b5, w.b7]
            # uni_lstm_b = [w.b0, w.b1, w.b2, w.b3, w.b4, w.b5, w.b6, w.b7]

            for uni_w, pre_w in zip(uni_lstm_w, source_w):
                uni_w.data[:] = pre_w.data[:]

            for uni_b, pre_b in zip(uni_lstm_b, source_b):
                uni_b.data[:] = pre_b.data[:]

    def set_train(self, train):
        self.train = train

    def freeze_word_emb(self):
        self.word_embed.W.update_rule.enabled = False


    def compute_all_nearest_words(self, top_k=15):
        # xp = np
        self.nearest_ids = None
        xp = self.xp
        vocab_size = self.vocab_size
        if self.args.use_limit_vocab:
            vocab_size = self.train_vocab_size
        word_embs = self.word_embed.W.data[:vocab_size]
        norm_word_embs_gpu = word_embs / self.xp.linalg.norm(word_embs, axis=1).reshape(-1, 1)
        norm_word_embs_gpu_T = norm_word_embs_gpu.T
        batchsize = self.args.batchsize_nn
        iteration_list = range(0, word_embs.shape[0], batchsize)
        score_list = []
        top_idx_list = []
        self.logging.info('start finding nearest words...')
        for index in iteration_list:
            emb = norm_word_embs_gpu[index:index + batchsize]
            scores = xp.dot(emb, norm_word_embs_gpu_T)
            top_idx = xp.argsort(-scores, axis=1)

            offsent = self.args.nn_k_offset
            if offsent >= 0:
                top_idx = top_idx[:, offsent:top_k + offsent]
            else:
                top_idx = top_idx[:, -top_k:]
            top_idx = to_cpu(top_idx)
            top_idx_list.append(top_idx)
        self.logging.info('[finish!]')

        nearest_ids = np.concatenate(top_idx_list, axis=0)
        nearest_ids = np.array(nearest_ids, dtype=np.int32)
        nearest_ids = to_gpu(nearest_ids)
        self.nearest_ids = nearest_ids
        return nearest_ids


    def get_nearest_words(self, x_data, noise=None, ign_offset=False):
        if self.args.online_nn == 0 and noise is None:
            return self.nearest_ids[x_data]
        top_k = self.args.nn_k
        xs_var = self.word_embed(x_data)
        if noise is not None:
            xs_var += noise
        xs_var = xs_var.data
        xs_var_norm = xs_var / self.xp.linalg.norm(xs_var, axis=1, keepdims=True)
        vocab_size = self.vocab_size
        if self.args.use_limit_vocab:
            vocab_size = self.train_vocab_size
        word_embs = self.word_embed.W.data[:vocab_size]
        word_embs_norm = word_embs / self.xp.linalg.norm(word_embs, axis=1, keepdims=True)
        scores = self.xp.dot(xs_var_norm, word_embs_norm.T)
        top_idx = self.xp.argsort(-scores, axis=1)
        offsent = self.args.nn_k_offset
        if offsent >= 0 and ign_offset is False:
            top_idx = top_idx[:, offsent:top_k + offsent]
        else:
            top_idx = top_idx[:, :top_k]
        return top_idx.astype(self.xp.int32)


    def output_mlp(self, hy):
        # hy = F.dropout(hy, ratio=self.use_dropout)
        hy = self.hidden_layer(hy)
        hy = F.relu(hy)
        hy = F.dropout(hy, ratio=self.use_dropout)
        output = self.output_layer(hy)
        return output

    def __call__(self, x_data, lengths=None, d=None, first_step=False):
        batchsize = len(x_data)
        h_shape = (self.n_layers, batchsize, self.hidden_dim)
        hx = None
        cx = None

        x_data = self.xp.concatenate(x_data, axis=0)
        xs = self.word_embed(x_data)
        # dropout
        xs = F.dropout(xs, ratio=self.use_dropout)

        adv_flag = self.train and (self.use_adv or self.args.use_semi_data)

        if adv_flag:

            def norm_vec_sentence_level(d, nn_flag=False, include_norm_term=False):
                dim = d.shape[1]
                d_list = F.split_axis(d, np.cumsum(lengths)[:-1], axis=0)
                max_length = np.max(lengths)
                d_pad = F.pad_sequence(d_list, length=max_length, padding=0.0)
                d_flat = F.reshape(get_normalized_vector(d_pad, None), (-1, dim))
                split_size = np.cumsum(np.full(batchsize, max_length))[:-1]
                d_list = F.split_axis(d_flat, split_size, axis=0)
                d_list = [_d[:_length] for _d, _length in zip(d_list, lengths)]
                d = F.concat(d_list, axis=0)
                return d

            if first_step:
                if self.args.use_attn_d:
                    # ours (iVAT or iAdv)
                    idx = self.xp.arange(xs.shape[0]).astype(self.xp.int32)
                    # Compute Nearest Neighbor
                    nearest_ids = self.get_nearest_words(x_data)
                    self.nearest_ids_local = nearest_ids
                    nn_words = self.word_embed(nearest_ids)
                    nn_words = F.dropout(nn_words, ratio=self.use_dropout)
                    self.xs = xs
                    xs_broad = F.reshape(xs, (xs.shape[0], 1, -1))
                    xs_broad = F.broadcast_to(xs_broad, nn_words.shape)
                    self.nn_words = nn_words
                    diff = nn_words - xs_broad
                    diff_data = diff.data
                    self.diff = diff
                    self.diff_data = diff_data
                    self.idx = idx
                    shape = (diff_data.shape[0], diff_data.shape[1], 1)
                    if self.args.use_semi_data:
                        # iVat
                        d_attn = self.xp.ones(shape, dtype='f') * (1.0 / self.args.nn_k)
                    else:
                        # iAdv
                        d_attn = self.xp.zeros(shape, dtype='f')

                    # TODO: Add random pertubation for iVAT
                    dir_normed = get_normalized_vector(diff, None, (2))
                    self.dir_normed = dir_normed
                    attention_d_var = Variable(d_attn.astype(self.xp.float32))
                    self.attention_d_var = attention_d_var
                    attention_d = attention_d_var
                    if self.xp.sum(attention_d.data) != 0.0:
                        attention_d = F.normalize(attention_d, axis=1)
                    attention_d = F.broadcast_to(attention_d, dir_normed.shape)
                    attention_d = F.sum(attention_d * dir_normed.data, axis=1)
                    # Normalize at word-level
                    d_var = get_normalized_vector(attention_d, None)
                    self.d_var = d_var
                    xs = xs + self.args.xi_var_first * d_var

                else:
                    # previous methods
                    if self.args.use_semi_data:
                        # Vat [Miyato et al., 2017]
                        d = self.xp.random.normal(size=xs.shape, dtype='f')
                    else:
                        # Adv [Miyato et al., 2017]
                        d = self.xp.zeros(xs.shape, dtype='f')
                    # Normalize at word-level
                    d = get_normalized_vector(d, self.xp)
                    d_var = Variable(d.astype(self.xp.float32))
                    self.d_var = d_var
                    xs = xs + self.args.xi_var_first * d_var

            elif d is not None:
                d_original = d.data if isinstance(d, Variable) else d
                if self.args.norm_sentence_level:
                    # Normalize at sentence-level
                    d_variable = norm_vec_sentence_level(d, include_norm_term=True)
                    d = d_variable.data
                else:
                    # Normalize at word-level
                    d = get_normalized_vector(d_original, self.xp)

                xs_noise_final = self.xi_var * d
                xs = xs + xs_noise_final

        split_size = np.cumsum(lengths)[:-1]
        xs_f = F.split_axis(xs, split_size, axis=0)

        hy_f, cy_f, ys_list = self.uni_lstm(hx=hx, cx=cx, xs=xs_f)

        hy = [_h[-1] for _h in ys_list]
        hy = F.concat(hy, axis=0)
        hy = F.reshape(hy, (batchsize, -1))
        self.hy = hy

        output = self.output_mlp(hy)
        return output
