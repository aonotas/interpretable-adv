#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import utils
import lm_nets

import random
import numpy as np
import pickle

import chainer
from chainer import cuda
from chainer import optimizers
import chainer.functions as F

import logging
logger = logging.getLogger(__name__)

chainer.config.use_cudnn = 'always'
to_cpu = chainer.cuda.to_cpu
to_gpu = chainer.cuda.to_gpu

from chainer import serializers
import nets
import lm_nets

def main():

    logging.basicConfig(
        format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
        level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--batchsize', dest='batchsize', type=int,
                        default=32, help='learning minibatch size')
    parser.add_argument('--batchsize_semi', dest='batchsize_semi', type=int,
                        default=64, help='learning minibatch size')
    parser.add_argument('--n_epoch', dest='n_epoch', type=int, default=30,
                        help='n_epoch')
    parser.add_argument('--pretrained_model', dest='pretrained_model',
                        type=str, default='', help='pretrained_model')
    parser.add_argument('--use_unlabled_to_vocab', dest='use_unlabled_to_vocab',
                        type=int, default=1, help='use_unlabled_to_vocab')
    parser.add_argument('--use_rational', dest='use_rational',
                        type=int, default=0, help='use_rational')
    parser.add_argument('--save_name', dest='save_name', type=str,
                        default='sentiment_model', help='save_name')
    parser.add_argument('--n_layers', dest='n_layers', type=int,
                        default=1, help='n_layers')
    parser.add_argument('--alpha', dest='alpha',
                        type=float, default=0.001, help='alpha')
    parser.add_argument('--alpha_decay', dest='alpha_decay',
                        type=float, default=0.0, help='alpha_decay')
    parser.add_argument('--clip', dest='clip',
                        type=float, default=5.0, help='clip')
    parser.add_argument('--debug_mode', dest='debug_mode',
                        type=int, default=0, help='debug_mode')
    parser.add_argument('--use_exp_decay', dest='use_exp_decay',
                        type=int, default=1, help='use_exp_decay')
    parser.add_argument('--load_trained_lstm', dest='load_trained_lstm',
                        type=str, default='', help='load_trained_lstm')
    parser.add_argument('--freeze_word_emb', dest='freeze_word_emb',
                        type=int, default=0, help='freeze_word_emb')
    parser.add_argument('--dropout', dest='dropout',
                        type=float, default=0.50, help='dropout')
    parser.add_argument('--use_adv', dest='use_adv',
                        type=int, default=0, help='use_adv')
    parser.add_argument('--xi_var', dest='xi_var',
                        type=float, default=1.0, help='xi_var')
    parser.add_argument('--xi_var_first', dest='xi_var_first',
                        type=float, default=1.0, help='xi_var_first')
    parser.add_argument('--lower', dest='lower',
                        type=int, default=0, help='lower')
    parser.add_argument('--nl_factor', dest='nl_factor', type=float,
                        default=1.0, help='nl_factor')
    parser.add_argument('--min_count', dest='min_count', type=int,
                        default=1, help='min_count')
    parser.add_argument('--ignore_unk', dest='ignore_unk', type=int,
                        default=0, help='ignore_unk')
    parser.add_argument('--use_semi_data', dest='use_semi_data',
                        type=int, default=0, help='use_semi_data')
    parser.add_argument('--add_labeld_to_unlabel', dest='add_labeld_to_unlabel',
                        type=int, default=1, help='add_labeld_to_unlabel')
    parser.add_argument('--norm_sentence_level', dest='norm_sentence_level',
                        type=int, default=1, help='norm_sentence_level')
    parser.add_argument('--dataset', default='imdb',
                        choices=['imdb', 'elec', 'rotten', 'dbpedia', 'rcv1'])
    parser.add_argument('--eval', dest='eval', type=int, default=0, help='eval')
    parser.add_argument('--emb_dim', dest='emb_dim', type=int,
                        default=256, help='emb_dim')
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int,
                        default=1024, help='hidden_dim')
    parser.add_argument('--hidden_cls_dim', dest='hidden_cls_dim', type=int,
                        default=30, help='hidden_cls_dim')
    parser.add_argument('--adaptive_softmax', dest='adaptive_softmax',
                        type=int, default=1, help='adaptive_softmax')
    parser.add_argument('--random_seed', dest='random_seed', type=int,
                        default=1234, help='random_seed')
    parser.add_argument('--n_class', dest='n_class', type=int,
                        default=2, help='n_class')
    parser.add_argument('--word_only', dest='word_only', type=int,
                        default=0, help='word_only')
    # iVAT
    parser.add_argument('--use_attn_d', dest='use_attn_d',
        type=int, default=0, help='use_attn_d')
    parser.add_argument('--nn_k', dest='nn_k', type=int, default=10, help='nn_k')
    parser.add_argument('--nn_k_offset', dest='nn_k_offset',
        type=int, default=1, help='nn_k_offset')
    parser.add_argument('--online_nn', dest='online_nn',
        type=int, default=0, help='online_nn')
    parser.add_argument('--use_limit_vocab', dest='use_limit_vocab', type=int,
        default=1, help='use_limit_vocab')
    parser.add_argument('--batchsize_nn', dest='batchsize_nn',
        type=int, default=10, help='batchsize_nn')
    # Visualize
    parser.add_argument('--analysis_mode', dest='analysis_mode', type=int,
        default=0, help='analysis_mode')
    parser.add_argument('--analysis_limit', dest='analysis_limit', type=int,
        default=100, help='analysis_limit')

    args = parser.parse_args()
    batchsize = args.batchsize
    batchsize_semi = args.batchsize_semi
    print(args)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    os.environ["CHAINER_SEED"] = str(args.random_seed)
    os.makedirs("models", exist_ok=True)

    if args.debug_mode:
        chainer.set_debug(True)

    use_unlabled_to_vocab = args.use_unlabled_to_vocab
    lower = args.lower == 1
    n_char_vocab = 1
    n_class = 2
    if args.dataset == 'imdb':
        vocab_obj, dataset, lm_data, t_vocab = utils.load_dataset_imdb(
            include_pretrain=use_unlabled_to_vocab, lower=lower,
            min_count=args.min_count, ignore_unk=args.ignore_unk,
            use_semi_data=args.use_semi_data,
            add_labeld_to_unlabel=args.add_labeld_to_unlabel)
        (train_x, train_x_len, train_y,
         dev_x, dev_x_len, dev_y,
         test_x, test_x_len, test_y) = dataset
        vocab, vocab_count = vocab_obj
        n_class = 2
    # TODO: add other dataset code

    if args.use_semi_data:
        semi_train_x, semi_train_x_len = lm_data

    print('train_vocab_size:', t_vocab)

    vocab_inv = dict([(widx, w) for w, widx in vocab.items()])
    print('vocab_inv:', len(vocab_inv))

    xp = cuda.cupy if args.gpu >= 0 else np
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        xp.random.seed(args.random_seed)

    n_vocab = len(vocab)
    model = nets.uniLSTM_iVAT(n_vocab=n_vocab, emb_dim=args.emb_dim,
                             hidden_dim=args.hidden_dim,
                             use_dropout=args.dropout, n_layers=args.n_layers,
                             hidden_classifier=args.hidden_cls_dim,
                             use_adv=args.use_adv, xi_var=args.xi_var,
                             n_class=n_class, args=args)
    model.train_vocab_size = t_vocab
    model.vocab_size = n_vocab
    model.logging = logging

    if args.pretrained_model != '':
        # load pretrained LM model
        pretrain_model = lm_nets.RNNForLM(n_vocab, 1024, args.n_layers, 0.50,
                                          share_embedding=False,
                                          adaptive_softmax=args.adaptive_softmax)
        serializers.load_npz(args.pretrained_model, pretrain_model)
        pretrain_model.lstm = pretrain_model.rnn
        model.set_pretrained_lstm(pretrain_model, word_only=args.word_only)


    all_nn_flag = args.use_attn_d
    if all_nn_flag and args.online_nn == 0:
        word_embs = model.word_embed.W.data
        model.norm_word_embs = word_embs / np.linalg.norm(word_embs, axis=1).reshape(-1, 1)
        model.norm_word_embs = np.array(model.norm_word_embs, dtype=np.float32)

    if args.load_trained_lstm != '':
        serializers.load_hdf5(args.load_trained_lstm, model)

    if args.gpu >= 0:
        model.to_gpu()



    # Visualize mode
    if args.analysis_mode:
        def sort_statics(_x_len, name=''):
            sorted_len = sorted([(x_len, idx) for idx, x_len in enumerate(_x_len)], key=lambda x:x[0])
            return [idx for _len, idx in sorted_len]
        test_sorted = sort_statics(test_x_len, 'test')
        if args.analysis_limit > 0:
            test_sorted = test_sorted[:args.analysis_limit]

    if all_nn_flag and args.online_nn == 0:
        model.compute_all_nearest_words(top_k=args.nn_k)
        # check nearest words
        def most_sims(word):
            if word not in vocab:
                logging.info('[not found]:{}'.format(word))
                return False
            idx = vocab[word]
            idx_gpu = xp.array([idx], dtype=xp.int32)
            top_idx = model.get_nearest_words(idx_gpu)
            sim_ids = top_idx[0]
            words = [vocab_inv[int(i)] for i in sim_ids]
            word_line = ','.join(words)
            logging.info('{}\t\t{}'.format(word, word_line))

        most_sims(u'good')
        most_sims(u'this')
        most_sims(u'that')
        most_sims(u'awesome')
        most_sims(u'bad')
        most_sims(u'wrong')

    def evaluate(x_set, x_length_set, y_set):
        chainer.config.train = False
        chainer.config.enable_backprop = False
        iteration_list = range(0, len(x_set), batchsize)
        correct_cnt = 0
        total_cnt = 0.0
        predicted_np = []

        for i_index, index in enumerate(iteration_list):
            x = [to_gpu(_x) for _x in x_set[index:index + batchsize]]
            x_length = x_length_set[index:index + batchsize]
            y = to_gpu(y_set[index:index + batchsize])
            output = model(x, x_length)

            predict = xp.argmax(output.data, axis=1)
            correct_cnt += xp.sum(predict == y)
            total_cnt += len(y)

        accuracy = (correct_cnt / total_cnt) * 100.0
        chainer.config.enable_backprop = True
        return accuracy

    def get_unlabled(perm_semi, i_index):
        index = i_index * batchsize_semi
        sample_idx = perm_semi[index:index + batchsize_semi]
        x = [to_gpu(semi_train_x[_i]) for _i in sample_idx]
        x_length = [semi_train_x_len[_i] for _i in sample_idx]
        return x, x_length

    base_alpha = args.alpha
    opt = optimizers.Adam(alpha=base_alpha)
    opt.setup(model)
    opt.add_hook(chainer.optimizer.GradientClipping(args.clip))

    if args.freeze_word_emb:
        model.freeze_word_emb()

    prev_dev_accuracy = 0.0
    global_step = 0.0
    adv_rep_num_statics = {}
    adv_rep_pos_statics = {}

    if args.eval:
        dev_accuracy = evaluate(dev_x, dev_x_len, dev_y)
        log_str = ' [dev] accuracy:{}, length:{}'.format(str(dev_accuracy))
        logging.info(log_str)

        # test
        test_accuracy = evaluate(test_x, test_x_len, test_y)
        log_str = ' [test] accuracy:{}, length:{}'.format(str(test_accuracy))
        logging.info(log_str)


    for epoch in range(args.n_epoch):
        logging.info('epoch:' + str(epoch))
        # train
        model.cleargrads()
        chainer.config.train = True
        iteration_list = range(0, len(train_x), batchsize)


        if args.analysis_mode:
            # Visualize mode
            iteration_list = range(0, len(test_sorted), batchsize)
            chainer.config.train = False
            chainer.config.enable_backprop = True
            chainer.config.cudnn_deterministic = True
            chainer.config.use_cudnn = 'never'

        perm = np.random.permutation(len(train_x))
        if args.use_semi_data:
            perm_semi = [np.random.permutation(len(semi_train_x)) for _ in range(2)]
            perm_semi = np.concatenate(perm_semi, axis=0)
            # print 'perm_semi:', perm_semi.shape
        def idx_func(shape):
            return xp.arange(shape).astype(xp.int32)

        sum_loss = 0.0
        sum_loss_z = 0.0
        sum_loss_z_sparse = 0.0
        sum_loss_label = 0.0
        avg_rate = 0.0
        avg_rate_num = 0.0
        correct_cnt = 0
        total_cnt = 0.0
        N = len(iteration_list)
        is_adv_example_list = []
        is_adv_example_disc_list = []
        is_adv_example_disc_craft_list = []
        y_np = []
        predicted_np = []
        save_items = []
        vis_lists = []
        for i_index, index in enumerate(iteration_list):
            global_step += 1.0
            model.set_train(True)
            sample_idx = [test_sorted[i_index]]
            x = [to_gpu(test_x[_i]) for _i in sample_idx]
            x_length = [test_x_len[_i] for _i in sample_idx]

            y = to_gpu(test_y[sample_idx])



            d = None
            d_hidden = None

            # Classification loss
            output = model(x, x_length)
            output_original = output
            loss = F.softmax_cross_entropy(output, y, normalize=True)
            # Adversarial Training
            output = model(x, x_length, first_step=True, d=None)
            # Adversarial loss (First step)
            loss_adv_first = F.softmax_cross_entropy(output, y, normalize=True)
            model.cleargrads()
            loss_adv_first.backward()

            if args.use_attn_d:
                # iAdv
                attn_d_grad = model.attention_d_var.grad
                attn_d_grad = F.normalize(attn_d_grad, axis=1)
                # Get directional vector
                dir_normed = model.dir_normed.data
                attn_d = F.broadcast_to(attn_d_grad, dir_normed.shape).data
                d = xp.sum(attn_d * dir_normed, axis=1)
            else:
                # Adv
                d = model.d_var.grad
                attn_d_grad = chainer.Variable(d)
            d_data = d.data if isinstance(d, chainer.Variable) else d
            # sentence-normalize
            d_data = d_data / xp.linalg.norm(d_data)

            # Analysis mode
            predict_adv = xp.argmax(output.data, axis=1)
            predict = xp.argmax(output_original.data, axis=1)
            logging.info('predict:{}, gold:{}'.format(predict, y))

            x_concat = xp.concatenate(x, axis=0)

            is_wrong_predict = predict != y
            if is_wrong_predict:
                continue

            is_adv_example = predict_adv != y
            logging.info('is_adv_example:{}'.format(is_adv_example))
            is_adv_example = to_cpu(is_adv_example)

            idx = xp.arange(x_concat.shape[0]).astype(xp.int32)
            # compute Nearest Neighbor
            nearest_ids = model.get_nearest_words(x_concat)
            nn_words = model.word_embed(nearest_ids)
            nn_words = F.dropout(nn_words, ratio=args.dropout)
            xs = model.word_embed(x_concat)
            xs = F.dropout(xs, ratio=args.dropout)
            xs_broad = F.reshape(xs, (xs.shape[0], 1, -1))
            xs_broad = F.broadcast_to(xs_broad, nn_words.shape)
            diff = nn_words - xs_broad

            # compute similarity
            dir_normed = nets.get_normalized_vector(diff, None, (2)).data
            d_norm = nets.get_normalized_vector(d, xp)
            d_norm = xp.reshape(d_norm, (d_norm.shape[0], 1, -1))
            sims = F.matmul(dir_normed, d_norm, False, True)
            sims = xp.reshape(sims.data, (sims.shape[0], -1))

            most_sims_idx_top = xp.argsort(-sims, axis=1)[idx_func(sims.shape[0]), 0].reshape(-1)

            vis_items = []
            r_len = x[0].shape[0]
            for r_i in range(r_len):
                idx = r_i
                # most similar words in nearest neighbors
                max_sim_idx = most_sims_idx_top[idx]
                replace_word_idx = nearest_ids[idx, max_sim_idx]

                max_sim_scalar = xp.max(sims, axis=1)[idx].reshape(-1)
                attn_d_value = d_data[idx].reshape(-1)
                # grad_scale = xp.linalg.norm(d_data[idx]) / xp.max(xp.linalg.norm(d_data))
                grad_scale = xp.linalg.norm(d_data[idx]) / xp.max(xp.linalg.norm(d_data, axis=1))

                nn_words_list = [vocab_inv[int(n_i)] for n_i in nearest_ids[idx]]
                nn_words = ','.join(nn_words_list)

                sims_nn = sims[idx]


                diff_norm_scala = xp.linalg.norm(diff.data[idx, max_sim_idx])
                d_data_scala = xp.linalg.norm(d_data[idx])


                vis_item = [r_i, vocab_inv[int(x_concat[idx])], vocab_inv[int(replace_word_idx)],
                            to_cpu(max_sim_scalar), to_cpu(attn_d_value), nn_words, to_cpu(grad_scale), is_adv_example, to_cpu(sims_nn), to_cpu(diff_norm_scala), to_cpu(d_data_scala)]
                vis_items.append(vis_item)
            save_items.append([vis_items, to_cpu(x[0]), to_cpu(y)])

        with open(args.save_name, mode='wb') as f:
            # Save as pickle file
            pickle.dump(save_items, f, protocol=2)


if __name__ == '__main__':
    main()
