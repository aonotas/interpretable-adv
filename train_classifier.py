#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

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

import process_dataset
from chainer import serializers
import net
from sklearn.metrics import precision_recall_fscore_support, f1_score as f1_score_func

def main():

    logging.basicConfig(
        format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
        level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--batchsize', dest='batchsize', type=int,
                        default=64, help='learning minibatch size')
    parser.add_argument('--batchsize_semi', dest='batchsize_semi', type=int,
                        default=256, help='learning minibatch size')
    parser.add_argument('--n_epoch', dest='n_epoch', type=int, default=100, help='n_epoch')
    parser.add_argument('--pretrained_model', dest='pretrained_model',
                        type=str, default='', help='pretrained_model')
    parser.add_argument('--w2v_model', dest='w2v_model', type=str, default='', help='w2v_model')
    parser.add_argument('--w2v_norm', dest='w2v_norm', type=int, default=0, help='w2v_norm')
    parser.add_argument('--w2v_var', dest='w2v_var', type=float, default=0.0, help='w2v_var')
    parser.add_argument('--use_unlabled', dest='use_unlabled',
                        type=int, default=0, help='use_unlabled')
    parser.add_argument('--use_rational', dest='use_rational',
                        type=int, default=0, help='use_rational')
    parser.add_argument('--use_baseline', dest='use_baseline',
                        type=int, default=0, help='use_baseline')
    parser.add_argument('--use_baseline_attention', dest='use_baseline_attention',
                        type=int, default=0, help='use_baseline_attention')

    parser.add_argument('--save_name', dest='save_name', type=str,
                        default='sentiment_model_', help='save_name')
    parser.add_argument('--n_layers', dest='n_layers', type=int, default=1, help='n_layers')
    parser.add_argument('--generate_rule', dest='generate_rule',
                        type=int, default=0, help='generate_rule')
    parser.add_argument('--generate_rule_img', dest='generate_rule_img',
                        type=int, default=0, help='generate_rule_img')
    parser.add_argument('--salience_divid_count', dest='salience_divid_count',
                        type=int, default=0, help='salience_divid_count')
    parser.add_argument('--rule_word_num', dest='rule_word_num',
                        type=int, default=10, help='rule_word_num')
    parser.add_argument('--trained_model', dest='trained_model',
                        type=str, default='', help='trained_model')
    parser.add_argument('--rule_file', dest='rule_file',
                        type=str, default='', help='rule_file')
    parser.add_argument('--alpha', dest='alpha',
                        type=float, default=0.001, help='alpha')
    parser.add_argument('--alpha_decay', dest='alpha_decay',
                        type=float, default=0.0, help='alpha_decay')
    parser.add_argument('--clip', dest='clip',
                        type=float, default=5.0, help='clip')
    parser.add_argument('--l2', dest='l2',
                        type=float, default=0.0, help='l2')
    parser.add_argument('--nobias_lstm', dest='nobias_lstm',
                        type=int, default=0, help='nobias_lstm')
    parser.add_argument('--attenton_for_word', dest='attenton_for_word',
                        type=int, default=0, help='attenton_for_word')
    parser.add_argument('--num_attention', dest='num_attention',
                        type=int, default=1, help='num_attention')
    parser.add_argument('--debug', dest='debug',
                        type=int, default=0, help='debug')
    parser.add_argument('--debug_mode', dest='debug_mode',
                        type=int, default=0, help='debug_mode')
    parser.add_argument('--debug_sim', dest='debug_sim',
                        type=int, default=0, help='debug_sim')
    parser.add_argument('--debug_small', dest='debug_small',
                        type=int, default=0, help='debug_small')
    parser.add_argument('--word_only', dest='word_only',
                        type=int, default=0, help='word_only')
    parser.add_argument('--double_backward', dest='double_backward',
                        type=int, default=0, help='double_backward')
    parser.add_argument('--lambda_1', dest='lambda_1',
                        type=float, default=0.0002, help='lambda_1')
    parser.add_argument('--lambda_2', dest='lambda_2',
                        type=float, default=0.0004, help='lambda_2')
    parser.add_argument('--eps_decay', dest='eps_decay',
                        type=float, default=0.0001, help='eps_decay')
    parser.add_argument('--use_exp_decay', dest='use_exp_decay',
                        type=int, default=1, help='use_exp_decay')
    parser.add_argument('--debug_rational', dest='debug_rational',
                        type=int, default=0, help='debug_rational')
    parser.add_argument('--z_small_limit', dest='z_small_limit',
                        type=int, default=0, help='z_small_limit')
    parser.add_argument('--sampling', dest='sampling',
                        type=int, default=0, help='sampling')
    parser.add_argument('--load_trained_lstm', dest='load_trained_lstm',
                        type=str, default='', help='load_trained_lstm')
    parser.add_argument('--use_fast_lstm', dest='use_fast_lstm', type=int, default=1, help='1')
    parser.add_argument('--load_w_only', dest='load_w_only',
                        type=int, default=0, help='load_w_only')
    parser.add_argument('--freeze_word_emb', dest='freeze_word_emb',
                        type=int, default=0, help='freeze_word_emb')
    parser.add_argument('--dropout', dest='dropout',
                        type=float, default=0.50, help='dropout')
    parser.add_argument('--dot_cost_z', dest='dot_cost_z',
                        type=int, default=0, help='dot_cost_z')
    parser.add_argument('--use_z_word', dest='use_z_word',
                        type=int, default=0, help='use_z_word')
    parser.add_argument('--sep_loss', dest='sep_loss',
                        type=int, default=0, help='sep_loss')
    parser.add_argument('--use_salience', dest='use_salience',
                        type=int, default=0, help='use_salience')
    parser.add_argument('--lambda_1_upper', dest='lambda_1_upper',
                        type=int, default=0, help='lambda_1_upper')
    parser.add_argument('--use_rational_top', dest='use_rational_top',
                        type=int, default=0, help='use_rational_top')
    parser.add_argument('--num_rational', dest='num_rational',
                        type=int, default=10, help='num_rational')
    parser.add_argument('--length_decay', dest='length_decay',
                        type=int, default=0, help='length_decay')
    parser.add_argument('--use_just_norm', dest='use_just_norm',
                        type=int, default=0, help='use_just_norm')
    parser.add_argument('--norm_emb', dest='norm_emb',
                        type=int, default=0, help='norm_emb')
    parser.add_argument('--norm_emb_every', dest='norm_emb_every',
                        type=int, default=1, help='norm_emb_every')
    parser.add_argument('--use_adv', dest='use_adv',
                        type=int, default=0, help='use_adv')
    parser.add_argument('--xi_var', dest='xi_var',
                        type=float, default=1.0, help='xi_var')
    parser.add_argument('--xi_var_first', dest='xi_var_first',
                        type=float, default=1.0, help='xi_var_first')
    parser.add_argument('--lower', dest='lower',
                        type=int, default=1, help='lower')
    parser.add_argument('--use_adv_hidden', dest='use_adv_hidden',
                        type=int, default=0, help='use_adv_hidden')
    parser.add_argument('--predict_next', dest='predict_next',
                        type=int, default=0, help='predict_next')
    parser.add_argument('--use_adv_and_nl_loss', dest='use_adv_and_nl_loss',
                        type=int, default=1, help='use_adv_and_nl_loss')
    parser.add_argument('--norm_lambda', dest='norm_lambda',
                        type=float, default=1.0, help='norm_lambda')
    parser.add_argument('--nl_factor', dest='nl_factor', type=float, default=1.0, help='nl_factor')
    parser.add_argument('--bnorm', dest='bnorm', type=int, default=0, help='bnorm')
    parser.add_argument('--bnorm_hidden', dest='bnorm_hidden',
                        type=int, default=0, help='bnorm_hidden')
    parser.add_argument('--limit_length', dest='limit_length',
                        type=int, default=0, help='limit_length')
    parser.add_argument('--min_count', dest='min_count', type=int, default=1, help='min_count')
    parser.add_argument('--ignore_unk', dest='ignore_unk', type=int, default=0, help='ignore_unk')
    parser.add_argument('--nn_mixup', dest='nn_mixup', type=int, default=0, help='nn_mixup')
    parser.add_argument('--update_nearest_epoch', dest='update_nearest_epoch',
                        type=int, default=0, help='update_nearest_epoch')
    parser.add_argument('--mixup_lambda', dest='mixup_lambda',
                        type=float, default=0.5, help='mixup_lambda')
    parser.add_argument('--mixup_prob', dest='mixup_prob',
                        type=float, default=0.5, help='mixup_prob')
    parser.add_argument('--mixup_type', dest='mixup_type',
                        type=str, default='dir', help='mixup_type')
    parser.add_argument('--mixup_dim', dest='mixup_dim',
                        type=int, default=0, help='mixup_dim')
    parser.add_argument('--nn_k', dest='nn_k', type=int, default=15, help='nn_k')
    parser.add_argument('--nn_k_offset', dest='nn_k_offset',
                        type=int, default=1, help='nn_k_offset')
    parser.add_argument('--norm_mean_var', dest='norm_mean_var',
                        type=int, default=0, help='norm_mean_var')
    parser.add_argument('--word_drop', dest='word_drop', type=int, default=0, help='word_drop')
    parser.add_argument('--word_drop_prob', dest='word_drop_prob',
                        type=float, default=0.25, help='word_drop_prob')
    parser.add_argument('--fix_lstm_norm', dest='fix_lstm_norm',
                        type=int, default=0, help='fix_lstm_norm')
    parser.add_argument('--use_semi_data', dest='use_semi_data',
                        type=int, default=0, help='use_semi_data')
    parser.add_argument('--use_semi_vat', dest='use_semi_vat',
                        type=int, default=1, help='use_semi_vat')
    parser.add_argument('--use_semi_pred_adv', dest='use_semi_pred_adv',
                        type=int, default=0, help='use_semi_pred_adv')
    parser.add_argument('--use_af_dropout', dest='use_af_dropout',
                        type=int, default=0, help='use_af_dropout')
    parser.add_argument('--use_nn_term', dest='use_nn_term',
                        type=int, default=0, help='use_nn_term')
    parser.add_argument('--online_nn', dest='online_nn',
                        type=int, default=0, help='online_nn')
    parser.add_argument('--nn_type', dest='nn_type', type=str, default='dir', help='nn_type')
    parser.add_argument('--nn_div', dest='nn_div', type=int, default=1, help='nn_div')
    parser.add_argument('--xi_type', dest='xi_type',
                        type=str, default='fixed', help='xi_type')
    parser.add_argument('--batchsize_nn', dest='batchsize_nn',
                        type=int, default=10, help='batchsize_nn')
    parser.add_argument('--add_labeld_to_unlabel', dest='add_labeld_to_unlabel',
                        type=int, default=1, help='add_labeld_to_unlabel')
    parser.add_argument('--add_dev_to_unlabel', dest='add_dev_to_unlabel',
                        type=int, default=0, help='add_dev_to_unlabel')
    parser.add_argument('--add_fullvocab', dest='add_fullvocab',
                        type=int, default=0, help='add_fullvocab')
    parser.add_argument('--norm_freq', dest='norm_freq',
                        type=int, default=0, help='norm_freq')
    parser.add_argument('--save_flag', dest='save_flag',
                        type=int, default=1, help='save_flag')
    parser.add_argument('--norm_sentence_level', dest='norm_sentence_level',
                        type=int, default=0, help='norm_sentence_level')
    parser.add_argument('--eps_zeros', dest='eps_zeros',
                        type=int, default=0, help='eps_zeros')
    parser.add_argument('--eps_abs', dest='eps_abs',
                        type=int, default=0, help='eps_abs')
    parser.add_argument('--sampling_eps', dest='sampling_eps',
                        type=int, default=0, help='sampling_eps')
    parser.add_argument('--save_last', dest='save_last',
                        type=int, default=0, help='save_last')
    parser.add_argument('--norm_sent_noise', dest='norm_sent_noise',
                        type=int, default=0, help='norm_sent_noise')
    parser.add_argument('--freeze_nn', dest='freeze_nn',
                        type=int, default=0, help='freeze_nn')
    parser.add_argument('--vat_iter', dest='vat_iter',
                        type=int, default=1, help='vat_iter')
    parser.add_argument('--all_eps', dest='all_eps',
                        type=int, default=0, help='all_eps')
    parser.add_argument('--af_xi_var', dest='af_xi_var',
                        type=float, default=1.0, help='af_xi_var')
    parser.add_argument('--reverse_loss', dest='reverse_loss',
                        type=int, default=0, help='reverse_loss')
    parser.add_argument('--loss_eps', dest='loss_eps',
                        type=float, default=1.0, help='loss_eps')
    parser.add_argument('--init_d_with_nn', dest='init_d_with_nn',
                        type=int, default=0, help='init_d_with_nn')
    parser.add_argument('--ignore_fast_sent_norm', dest='ignore_fast_sent_norm',
                        type=int, default=0, help='ignore_fast_sent_norm')
    parser.add_argument('--ignore_norm', dest='ignore_norm',
                        type=int, default=0, help='ignore_norm')
    parser.add_argument('--init_d_type', dest='init_d_type',
                        type=str, default='rand_nn', help='init_d_type')
    parser.add_argument('--use_d_fixed', dest='use_d_fixed',
                        type=int, default=0, help='use_d_fixed')
    parser.add_argument('--nn_term_sq', dest='nn_term_sq',
                        type=int, default=0, help='nn_term_sq')
    parser.add_argument('--nn_term_sq_half', dest='nn_term_sq_half',
                        type=int, default=0, help='nn_term_sq_half')
    parser.add_argument('--init_d_noise', dest='init_d_noise',
                        type=int, default=0, help='init_d_noise')
    parser.add_argument('--eps_scale', dest='eps_scale',
                        type=float, default=1.0, help='eps_scale')
    parser.add_argument('--sim_type', dest='sim_type',
                        type=str, default='cos', help='sim_type')
    parser.add_argument('--use_all_diff', dest='use_all_diff',
                        type=int, default=1, help='use_all_diff')
    parser.add_argument('--use_first_avg', dest='use_first_avg',
                        type=int, default=0, help='use_first_avg')
    parser.add_argument('--use_norm_d', dest='use_norm_d',
                        type=int, default=1, help='use_norm_d')
    parser.add_argument('--eps_min', dest='eps_min', type=float, default=0.0, help='eps_min')
    parser.add_argument('--eps_max', dest='eps_max', type=float, default=0.0, help='eps_max')
    parser.add_argument('--eps_minus', dest='eps_minus', type=float, default=0.0, help='eps_minus')
    parser.add_argument('--use_random_nn', dest='use_random_nn',
                        type=int, default=0, help='use_random_nn')
    parser.add_argument('--use_attn_d', dest='use_attn_d',
                        type=int, default=0, help='use_attn_d')
    parser.add_argument('--use_softmax', dest='use_softmax',
                        type=int, default=0, help='use_softmax')
    parser.add_argument('--use_attn_full', dest='use_attn_full', type=int, default=0, help='use_attn_full')
    parser.add_argument('--use_attn_dot', dest='use_attn_dot', type=int, default=0, help='use_attn_dot')
    parser.add_argument('--dot_k', dest='dot_k', type=int, default=1, help='dot_k')
    parser.add_argument('--up_grad', dest='up_grad', type=int, default=0, help='up_grad')
    parser.add_argument('--up_grad_attn', dest='up_grad_attn', type=int, default=0, help='up_grad_attn')
    parser.add_argument('--no_grad', dest='no_grad', type=int, default=0, help='no_grad')
    parser.add_argument('--use_nn_drop', dest='use_nn_drop',
                        type=int, default=0, help='use_nn_drop')
    parser.add_argument('--use_onehot', dest='use_onehot',
                        type=int, default=0, help='use_onehot')
    parser.add_argument('--init_rand_diff_d', dest='init_rand_diff_d',
                        type=int, default=0, help='init_rand_diff_d')
    parser.add_argument('--norm_diff', dest='norm_diff', type=int, default=0, help='norm_diff')
    parser.add_argument('--norm_diff_sent', dest='norm_diff_sent', type=int, default=0, help='norm_diff_sent')
    parser.add_argument('--norm_diff_sent_first', dest='norm_diff_sent_first', type=int, default=0, help='norm_diff_sent_first')
    parser.add_argument('--auto_scale_eps', dest='auto_scale_eps',
                        type=int, default=0, help='auto_scale_eps')
    parser.add_argument('--use_concat_random_ids', dest='use_concat_random_ids',
                        type=int, default=0, help='use_concat_random_ids')
    parser.add_argument('--use_d_original_most_sim', dest='use_d_original_most_sim',
                        type=int, default=0, help='use_d_original_most_sim')
    parser.add_argument('--use_important_score', dest='use_important_score',
                        type=int, default=0, help='use_important_score')
    parser.add_argument('--eps_zeros_minus', dest='eps_zeros_minus', type=int, default=0, help='eps_zeros_minus')
    parser.add_argument('--use_attn_drop', dest='use_attn_drop', type=int, default=0, help='use_attn_drop')
    parser.add_argument('--imp_type', dest='imp_type', type=int, default=0, help='imp_type')
    parser.add_argument('--eps_diff', dest='eps_diff', type=int, default=0, help='eps_diff')
    parser.add_argument('--use_limit_vocab', dest='use_limit_vocab', type=int, default=0, help='use_limit_vocab')
    parser.add_argument('--use_plus_d', dest='use_plus_d', type=int, default=0, help='use_plus_d')
    parser.add_argument('--double_adv', dest='double_adv', type=int, default=0, help='double_adv')
    parser.add_argument('--init_d_adv', dest='init_d_adv', type=int, default=0, help='init_d_adv')
    parser.add_argument('--adv_mode', dest='adv_mode', type=int, default=0, help='adv_mode')
    parser.add_argument('--analysis_mode', dest='analysis_mode', type=int, default=0, help='analysis_mode')
    parser.add_argument('--scala_plus', dest='scala_plus', type=int, default=0, help='scala_plus')
    parser.add_argument('--use_plus_zeros', dest='use_plus_zeros', type=int, default=0, help='use_plus_zeros')
    parser.add_argument('--use_attn_one', dest='use_attn_one', type=int, default=0, help='use_attn_one')
    parser.add_argument('--init_d_with', dest='init_d_with', type=float, default=0.0, help='init_d_with')
    parser.add_argument('--kmeans', dest='kmeans', type=int, default=0, help='kmeans')
    parser.add_argument('--n_clusters', dest='n_clusters', type=int, default=100, help='n_clusters')
    parser.add_argument('--top_filter_rate', dest='top_filter_rate', type=float, default=0.10, help='top_filter_rate')
    parser.add_argument('--freeze_d_plus', dest='freeze_d_plus', type=int, default=0, help='freeze_d_plus')
    parser.add_argument('--init_grad', dest='init_grad', type=int, default=0, help='init_grad')
    parser.add_argument('--init_d_attn_ones', dest='init_d_attn_ones', type=int, default=0, help='init_d_attn_ones')
    parser.add_argument('--init_d_fact', dest='init_d_fact', type=float, default=1.0, help='init_d_fact')
    parser.add_argument('--init_d_random', dest='init_d_random', type=int, default=0, help='init_d_random')
    parser.add_argument('--norm_diff_all', dest='norm_diff_all', type=int, default=0, help='norm_diff_all')
    parser.add_argument('--norm_sent_attn_scala', dest='norm_sent_attn_scala', type=int, default=0, help='norm_sent_attn_scala')
    parser.add_argument('--rep_sim_noise_word', dest='rep_sim_noise_word', type=int, default=0, help='rep_sim_noise_word')
    parser.add_argument('--ign_noise_eos', dest='ign_noise_eos', type=int, default=0, help='ign_noise_eos')
    parser.add_argument('--search_iters', dest='search_iters', type=int, default=10, help='search_iters')
    parser.add_argument('--adv_type', dest='adv_type', type=int, default=0, help='adv_type')
    parser.add_argument('--use_saliency', dest='use_saliency', type=int, default=0, help='use_saliency')
    parser.add_argument('--adv_iter', dest='adv_iter', type=int, default=1, help='adv_iter')
    parser.add_argument('--fixed_d', dest='fixed_d', type=int, default=0, help='fixed_d')
    parser.add_argument('--max_attn', dest='max_attn', type=int, default=0, help='max_attn')
    parser.add_argument('--max_attn_type', dest='max_attn_type', type=int, default=0, help='max_attn_type')
    parser.add_argument('--use_grad_scale', dest='use_grad_scale', type=int, default=0, help='use_grad_scale')
    parser.add_argument('--scale_type', dest='scale_type', type=int, default=0, help='scale_type')
    parser.add_argument('--print_info', dest='print_info', type=int, default=0, help='print_info')
    parser.add_argument('--soft_int', dest='soft_int', type=float, default=1.0, help='soft_int')
    parser.add_argument('--soft_int_final', dest='soft_int_final', type=float, default=1.0, help='soft_int_final')
    parser.add_argument('--adv_mode_iter', dest='adv_mode_iter', type=int, default=100, help='adv_mode_iter')
    parser.add_argument('--ignore_norm_final', dest='ignore_norm_final', type=int, default=0, help='ignore_norm_final')
    parser.add_argument('--noise_factor', dest='noise_factor', type=float, default=1.0, help='noise_factor')
    parser.add_argument('--use_zero_d', dest='use_zero_d', type=int, default=0, help='use_zero_d')
    parser.add_argument('--use_random_max', dest='use_random_max', type=int, default=0, help='use_random_max')
    parser.add_argument('--analysis_limit', dest='analysis_limit', type=int, default=0, help='analysis_limit')

    parser.add_argument('--use_weight_alpha', dest='use_weight_alpha', type=int, default=0, help='use_weight_alpha')
    parser.add_argument('--weight_type', dest='weight_type', type=int, default=0, help='weight_type')


    parser.add_argument('--dataset', default='imdb',
                        choices=['imdb', 'elec', 'rotten', 'dbpedia', 'rcv1', 'conll_2014', 'fce','ptb', 'wikitext-2', 'wikitext-103'])
    parser.add_argument('--use_seq_labeling', dest='use_seq_labeling', type=int, default=0, help='use_seq_labeling')
    parser.add_argument('--use_seq_labeling_pickle', dest='use_seq_labeling_pickle', type=int, default=0, help='use_seq_labeling_pickle')
    parser.add_argument('--use_bilstm', dest='use_bilstm', type=int, default=0, help='use_bilstm')
    parser.add_argument('--use_bilstm_forget', dest='use_bilstm_forget', type=int, default=0, help='use_bilstm_forget')
    parser.add_argument('--use_crf', dest='use_crf', type=int, default=0, help='use_crf')
    parser.add_argument('--use_all_for_lm', dest='use_all_for_lm', type=int, default=0, help='use_all_for_lm')
    parser.add_argument('--accuracy_for_wrong', dest='accuracy_for_wrong', type=int, default=0, help='accuracy_for_wrong')
    parser.add_argument('--accuracy_sentence', dest='accuracy_sentence', type=int, default=0, help='accuracy_sentence')
    parser.add_argument('--accuracy_f1', dest='accuracy_f1', type=int, default=0, help='accuracy_f1')
    parser.add_argument('--use_char', dest='use_char', type=int, default=0, help='use_char')


    parser.add_argument('--cs', dest='cs', type=int, default=0, help='cs')
    parser.add_argument('--debug_eval', dest='debug_eval', type=int, default=0, help='debug_eval')
    parser.add_argument('--emb_dim', dest='emb_dim', type=int, default=256, help='emb_dim')
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=1024, help='hidden_dim')
    parser.add_argument('--hidden_cls_dim', dest='hidden_cls_dim', type=int, default=30, help='hidden_cls_dim')
    parser.add_argument('--adaptive_softmax', dest='adaptive_softmax', type=int, default=1, help='adaptive_softmax')
    parser.add_argument('--use_w2v_flag', dest='use_w2v_flag', type=int, default=0, help='use_w2v_flag')
    parser.add_argument('--sent_loss', dest='sent_loss', type=int, default=0, help='sent_loss')
    parser.add_argument('--sent_loss_usual', dest='sent_loss_usual', type=int, default=0, help='sent_loss_usual')
    parser.add_argument('--use_ortho', dest='use_ortho', type=int, default=0, help='use_ortho')

    parser.add_argument('--analysis_loss', dest='analysis_loss', type=int, default=0, help='analysis_loss')
    parser.add_argument('--analysis_data', dest='analysis_data', type=int, default=2, help='analysis_data')
    parser.add_argument('--fil_type', dest='fil_type', type=int, default=0, help='fil_type')
    parser.add_argument('--analysis_mode_type', dest='analysis_mode_type', type=int, default=0, help='analysis_mode_type')
    parser.add_argument('--tsne_mode', dest='tsne_mode', type=int, default=0, help='tsne_mode')
    parser.add_argument('--bar_mode', dest='bar_mode', type=int, default=0, help='bar_mode')
    parser.add_argument('--attentional_d_mode', dest='attentional_d_mode', type=int, default=0, help='attentional_d_mode')
    parser.add_argument('--random_noise', dest='random_noise', type=int, default=0, help='random_noise')
    parser.add_argument('--random_noise_vat', dest='random_noise_vat', type=int, default=0, help='random_noise_vat')
    parser.add_argument('--div_attn_d', dest='div_attn_d', type=int, default=0, help='div_attn_d')

    parser.add_argument('--use_attn_sent_norm', dest='use_attn_sent_norm', type=int, default=0, help='use_attn_sent_norm')
    parser.add_argument('--random_seed', dest='random_seed', type=int, default=1234, help='random_seed')

    parser.add_argument('--n_class', dest='n_class', type=int, default=2, help='n_class')
    parser.add_argument('--init_scale', dest='init_scale', type=float, default=0.0, help='init_scale')
    parser.add_argument('--train_test_flag', dest='train_test_flag', type=int, default=0, help='train_test_flag')


    # TODO: alpha decayのタイミング (minを設定する？毎回？1 epochごと？)
    # TODO: decay はexponential decayと言って, 指数的に減衰させる方法

    # TODO: VATやAdvのノルムの大きさを自動選択させたい. nearestのpointまでの大きさをmaximumにさせる？
    # TODO: VATで選択するのはnearest nnを選択させるだけにする？
    # TODO: AdvExampleをPretrained LMから作る
    # TODO: hiddenに対してadvを入れる
    # TODO: 中心点のtermを追加する

    # TODO: UNKは存在しない (vocabに存在しないワードは無視する)
    # TODO: frequency 1以下のものは削除する (EOS)は頻度１とする

    args = parser.parse_args()
    batchsize = args.batchsize
    batchsize_semi = args.batchsize_semi
    print(args)


    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    os.environ["CHAINER_SEED"] = str(args.random_seed)

    if args.debug_mode:
        chainer.set_debug(True)

    # NOTE: use large vocab here
    # use_pretrain = args.pretrained_model != ''
    use_unlabled = args.use_unlabled
    lower = args.lower == 1
    print('lower:', lower)
    n_char_vocab = 1
    n_class = args.n_class

    if args.dataset == 'imdb':
        vocab_obj, dataset, lm_dataset, train_vocab_size = process_dataset.load_dataset(
            include_pretrain=use_unlabled, lower=lower, min_count=args.min_count,
            ignore_unk=args.ignore_unk, return_count=True, use_semi_data=args.use_semi_data,
            add_labeld_to_unlabel=args.add_labeld_to_unlabel, add_dev_to_unlabel=args.add_dev_to_unlabel, add_fullvocab=args.add_fullvocab)
        (train_x, train_x_len, train_y,
         dev_x, dev_x_len, dev_y,
         test_x, test_x_len, test_y) = dataset
        vocab, vocab_count = vocab_obj

    elif args.dataset == 'elec':
        vocab_obj, dataset, lm_dataset, train_vocab_size = process_dataset.load_dataset_elec(
            include_pretrain=use_unlabled, lower=lower, min_count=args.min_count,
            ignore_unk=args.ignore_unk, return_count=True, use_semi_data=args.use_semi_data,
            add_labeld_to_unlabel=args.add_labeld_to_unlabel, add_dev_to_unlabel=args.add_dev_to_unlabel, add_fullvocab=args.add_fullvocab)
        (train_x, train_x_len, train_y,
         dev_x, dev_x_len, dev_y,
         test_x, test_x_len, test_y) = dataset
        vocab, vocab_count = vocab_obj

    elif args.dataset == 'rotten':
        lower = True # TODO: check lower or not
        min_count = 1
        ignore_unk = 1
        vocabs, dataset, lm_dataset, train_vocab_size = process_dataset.load_dataset_rotten(
            include_pretrain=True, lower=lower, min_count=min_count,
            ignore_unk=ignore_unk, return_count=True,
            add_dev_to_unlabel=False, random_seed=args.random_seed, use_semi_data=args.use_semi_data)
        vocab, vocab_count = vocabs
        (lm_train_dataset, lm_dev_dataset) = lm_dataset
        (train_x, train_x_len, train_y,
         dev_x, dev_x_len, dev_y,
         test_x, test_x_len, test_y) = dataset
        n_vocab = len(vocab)

    elif args.dataset == 'rcv1':
        lower = True # TODO: check lower or not
        min_count = 1
        ignore_unk = 1
        vocabs, dataset, lm_dataset, train_vocab_size = process_dataset.load_dataset_rcv1(
            include_pretrain=True, lower=lower, min_count=min_count,
            ignore_unk=ignore_unk, return_count=True,
            add_dev_to_unlabel=False, use_semi_data=args.use_semi_data, train_test_flag=args.train_test_flag)
        vocab, vocab_count = vocabs
        # (lm_train_dataset, lm_dev_dataset) = lm_dataset
        # (lm_train_dataset, lm_dev_dataset) = lm_dataset
        # del lm_dataset
        (train_x, train_x_len, train_y,
         dev_x, dev_x_len, dev_y,
         test_x, test_x_len, test_y) = dataset
        n_vocab = len(vocab)
        logging.info('n_vocab:{}'.format(n_vocab))
        logging.info('train_vocab_size:{}'.format(train_vocab_size))

    elif args.dataset == 'dbpedia':
        lower = True # TODO: check lower or not
        min_count = 1
        ignore_unk = 1
        vocabs, dataset, lm_dataset, train_vocab_size = process_dataset.load_dataset_dbpedia(
            include_pretrain=True, lower=lower, min_count=min_count,
            ignore_unk=ignore_unk, return_count=True,
            add_dev_to_unlabel=0)
        vocab, vocab_count = vocabs
        (lm_train_dataset, lm_dev_dataset) = lm_dataset
        # del lm_dataset
        (train_x, train_x_len, train_y,
         dev_x, dev_x_len, dev_y,
         test_x, test_x_len, test_y) = dataset
        # train = lm_train_dataset[:]
        # val = lm_dev_dataset[:]
        # test = lm_dev_dataset[:]
        n_vocab = len(vocab)
    elif args.dataset == 'fce':
        vocab, doc_counts, dataset, lm_dataset, w2v = process_dataset.load_fce(lower=args.lower, min_count=args.min_count, ignore_unk=False, use_w2v_flag=args.use_w2v_flag, use_semi_data=args.use_semi_data)
        (train_x, train_x_len, train_y,
         dev_x, dev_x_len, dev_y,
         test_x, test_x_len, test_y) = dataset
        vocab_count = doc_counts
        train_vocab_size = len(vocab)
        # print 'train_x:', len(train_x)
        # print 'dev_x:', len(dev_x)
        # print 'test_x:', len(test_x)
        lm_dataset = (train_x, train_x_len)
        # semi_train_x, semi_train_x_len = lm_dataset

    elif args.dataset == 'conll_2014':


        # vocab_obj, dataset, lm_dataset, train_vocab_size = process_dataset.load_dataset(
        #     include_pretrain=use_unlabled, lower=lower, min_count=args.min_count,
        #     ignore_unk=args.ignore_unk, return_count=True, use_semi_data=args.use_semi_data,
        #     add_labeld_to_unlabel=args.add_labeld_to_unlabel, add_dev_to_unlabel=args.add_dev_to_unlabel, add_fullvocab=args.add_fullvocab)
        # (train_x, train_x_len, train_y,
        #  dev_x, dev_x_len, dev_y,
        #  test_x, test_x_len, test_y) = dataset
        # vocab, vocab_count = vocab_obj

        result = process_dataset.load_conll2014(lower=False, min_count=2, ignore_unk=False, use_all_for_lm=args.use_all_for_lm, use_char=args.use_char)
        if args.use_char:
            result_tmp, char_dataset = result
            vocab, vocab_count, dataset, correct_dataset, lm_train_dataset, lm_dev_dataset = result_tmp

            (vocab_char, train_x_char, train_x_char_len,
                            dev_x_char, dev_x_char_len,
                            test_x_char, test_x_char_len) = char_dataset
            n_char_vocab = len(vocab_char)

        else:
            vocab, vocab_count, dataset, correct_dataset, lm_train_dataset, lm_dev_dataset = result


        (train_x, train_x_len, train_y,
         dev_x, dev_x_len, dev_y,
         test_x, test_x_len, test_y) = dataset
        correct_train_x, correct_dev_x, correct_test_x = correct_dataset
        n_vocab = len(vocab)
        train_vocab_size = len(vocab)
        # print 'train_x:', len(train_x)
        # print 'dev_x:', len(dev_x)
        # print 'test_x:', len(test_x)


    if args.use_semi_data:
        semi_train_x, semi_train_x_len = lm_dataset
        # print 'semi_train_x:', len(semi_train_x)
        # print 'semi_train_x_len:', len(semi_train_x_len)
    print('train_vocab_size:', train_vocab_size)

    if args.limit_length:
        len_limit = 400
        train_x = [x[-len_limit:] for x in train_x]
        train_x_len = [x_len if x_len < len_limit else len_limit for x_len in train_x_len]
        dev_x = [x[-len_limit:] for x in dev_x]
        dev_x_len = [x_len if x_len < len_limit else len_limit for x_len in dev_x_len]
        test_x = [x[-len_limit:] for x in test_x]
        test_x_len = [x_len if x_len < len_limit else len_limit for x_len in test_x_len]

    if args.analysis_mode or args.analysis_mode_type==1:
        def sort_statics(_x_len, name=''):
            reverse_flag = True if args.use_seq_labeling == 1 else False
            sorted_len = sorted([(x_len, idx) for idx, x_len in enumerate(_x_len)], key=lambda x:x[0], reverse=reverse_flag)
            # print name, '(sorted):', sorted_len[:100]
            return [idx for _len, idx in sorted_len]

        # print 'train:', len(train_x_len)
        # print 'dev:', len(dev_x_len)
        # print 'test:', len(test_x_len)
        train_sorted = sort_statics(train_x_len, 'train')
        dev_sorted = sort_statics(dev_x_len, 'dev')
        test_sorted = sort_statics(test_x_len, 'test')
        # print 'test_sorted:', len(test_sorted)
        if args.analysis_limit > 0:
            test_sorted = test_sorted[:args.analysis_limit]

        if args.use_seq_labeling:
            analysis_dataset, analysis_correct_dataset = process_dataset.load_replace_text(vocab)

            (train_x, train_x_len, train_y,
                       dev_x, dev_x_len, dev_y,
                       test_x, test_x_len, test_y) = analysis_dataset
            (a_correct_train_x, a_correct_dev_x, a_correct_test_x) = analysis_correct_dataset

            if args.analysis_data == 0:
                test_x = train_x[:]
                test_x_len = train_x_len[:]
                test_y = train_y[:]
                correct_x = a_correct_train_x[:]
            elif args.analysis_data == 1:
                test_x = dev_x[:]
                test_x_len = dev_x_len[:]
                test_y = dev_y[:]
                correct_x = a_correct_dev_x[:]
            elif args.analysis_data == 2:
                # test_x = dev_x[:]
                # test_x_len = dev_x_len[:]
                # test_y = dev_y[:]
                correct_x = a_correct_test_x[:]




            # test_sorted = [idx for idx in range(len(test_x))]

            # test_x = a_test_x[:]


    vocab_inv = dict([(widx, w) for w, widx in vocab.items()])
    print('vocab_inv:', len(vocab_inv))

    xp = cuda.cupy if args.gpu >= 0 else np
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        xp.random.seed(1234)

    if args.debug_small:
        idx = range(len(train_x))
        random.shuffle(idx)
        idx = idx[:1000]
        idx = np.array(idx, dtype=np.int32)
        train_x = [train_x[i] for i in idx]
        train_x_len = [train_x_len[i] for i in idx]
        train_y = train_y[idx]

    n_vocab = len(vocab)
    if args.use_baseline == 1:
        model = net.BaselineLSTM(n_vocab=n_vocab, emb_dim=args.emb_dim, hidden_dim=args.hidden_dim,
                                 init_emb=None, use_dropout=args.dropout, n_layers=args.n_layers,
                                 hidden_classifier=args.hidden_cls_dim, nobias_lstm=args.nobias_lstm,
                                 use_my_lstm=args.generate_rule, norm_emb=args.norm_emb,
                                 use_adv=args.use_adv, xi_var=args.xi_var, n_char_vocab=n_char_vocab, n_class=args.n_class, args=args)
    elif args.use_baseline_attention == 1:
        model = net.AttentionBaselineLSTM(n_vocab=n_vocab, emb_dim=args.emb_dim, hidden_dim=args.hidden_dim,
                                          init_emb=None, use_dropout=args.dropout, n_layers=1,
                                          hidden_classifier=30, nobias_lstm=args.nobias_lstm,
                                          attenton_for_word=args.attenton_for_word,
                                          num_attention=args.num_attention)
    else:
        model = net.SentenceBiLSTM(n_vocab=n_vocab, emb_dim=args.emb_dim, hidden_dim=512,
                                   init_emb=None, use_dropout=args.dropout, n_layers=1)
    model.train_vocab_size = train_vocab_size
    model.vocab_size = n_vocab
    if args.pretrained_model != '':
        # pretrain_model = args.pretrained_model
        # load pretrain file
        # pre_n_vocab = 120795
        if 'best.model' in args.pretrained_model:
            import sys
            sys.path.append('fast_language_modeling')
            from fast_language_modeling import nets as lm_nets
            pretrain_model = lm_nets.RNNForLM(n_vocab, 1024, args.n_layers, 0.50,
                                              share_embedding=False,
                                              blackout_counts=None,
                                              adaptive_softmax=args.adaptive_softmax)
            # serializers.load_hdf5(args.load_trained_lstm, trained_lstm)
            serializers.load_npz(args.pretrained_model, pretrain_model)
            pretrain_model.lstm = pretrain_model.rnn
            model.set_pretrained_lstm(pretrain_model, word_only=args.word_only, is_L_lstm=0)

        elif 'fast_' in args.pretrained_model:
            counts = [float(1)
                      for w, idx in sorted(vocab.items(), key=lambda x: x[1])]
            pretrain_model = net.RNNForLMFast(n_vocab, n_word_emb=args.emb_dim, n_units=args.hidden_dim,
                                              counts=counts, sample_size=20)
            serializers.load_hdf5(args.pretrained_model, pretrain_model)
            model.set_pretrained_lstm(pretrain_model, word_only=args.word_only, is_L_lstm=1)

        else:
            pretrain_model = net.RNNForLM(n_vocab, n_word_emb=args.emb_dim, n_units=args.hidden_dim)
            serializers.load_hdf5(args.pretrained_model, pretrain_model)
            model.set_pretrained_lstm(pretrain_model, word_only=args.word_only, is_L_lstm=1)

    if args.load_trained_lstm != '':
        serializers.load_hdf5(args.load_trained_lstm, model)

    if args.w2v_model != '' or args.use_w2v_flag == 1 or args.use_w2v_flag == -1 or args.use_w2v_flag == 2:
        from gensim.models.keyedvectors import KeyedVectors
        w2v_model_path = args.w2v_model
        if args.use_w2v_flag == 1 or args.use_w2v_flag == -1 or args.use_w2v_flag == 2:
            pass
        else:
            w2v = KeyedVectors.load_word2vec_format(w2v_model_path, binary=True)
        bf_norm = np.average(np.linalg.norm(model.word_embed.W.data, axis=1))
        if args.w2v_norm:
            w2v.syn0 = (w2v.syn0 - np.mean(w2v.syn0, axis=1)
                        [..., None]) / np.std(w2v.syn0, axis=1)[..., None]
        if args.w2v_var > 0.0:
            w2v.syn0 = args.w2v_var * w2v.syn0
        pretrained_vecs = []
        for idx, w in sorted(vocab_inv.items(), key=lambda x: x[0]):
            # if w == '<eos>' and w not in w2v:
            #     w = '</s>'
            # if w == '<unk>':
            #     w = '</s>'
            if w in w2v:
                vec = w2v[w]
                model.word_embed.W.data[idx] = vec[:]
            else:
                # print 'notfound:', w
                vec = model.word_embed.W.data[idx]
        norm = np.linalg.norm(model.word_embed.W.data, axis=1).reshape(-1, 1)
        print('w2v norm:', norm)
        if args.fix_lstm_norm:
            af_norm = np.average(np.linalg.norm(model.word_embed.W.data, axis=1))
            model.fix_norm(bf_norm, af_norm)

    model.logging = logging

    all_nn_flag = (args.nn_mixup or args.use_nn_term or args.init_d_with_nn or args.use_attn_d)
    if all_nn_flag and args.online_nn == 0:
        word_embs = model.word_embed.W.data
        model.norm_word_embs = word_embs / np.linalg.norm(word_embs, axis=1).reshape(-1, 1)
        model.norm_word_embs = np.array(model.norm_word_embs, dtype=np.float32)

    if args.norm_freq or args.norm_mean_var:
        vocab_freq = np.array([float(vocab_count.get(w, 1)) for w, idx in
                               sorted(vocab.items(), key=lambda x: x[1])], dtype=np.float32)
        vocab_freq = vocab_freq / np.sum(vocab_freq)
        vocab_freq = vocab_freq.astype(np.float32)
        vocab_freq = vocab_freq[..., None]
        model.word_embed.vocab_freq = to_gpu(vocab_freq)

    # if args.norm_mean_var:
    #     # TODO: use word-quency
    #     # word_embs = model.word_embed.W.data
    #     # word_embs_norm = (word_embs - np.mean(word_embs, axis=1)
    #     #                   [..., None]) / np.std(word_embs, axis=1)[..., None]
    #     # model.word_embed.W.data[:] = word_embs_norm
    #     print('#norm_vecs')
    #     vocab_freqs = np.array([float(vocab_count.get(w, 1)) for w, idx in
    #                             sorted(vocab.items(), key=lambda x: x[1])], dtype=np.float32)
    #     vocab_freqs = vocab_freqs / np.sum(vocab_freqs)
    #     vocab_freqs = vocab_freqs.astype(np.float32)
    #     freq = vocab_freqs[..., None]
    #     print('freq:')
    #     print(freq)
    #     print('#norm_vecs...')
    #     word_embs = model.word_embed.W.data
    #     print('norm(word_embs):')
    #     print(np.linalg.norm(word_embs, axis=1).reshape(-1, 1))
    #     mean = np.sum(freq * word_embs, axis=0)
    #     print('mean:{}'.format(mean.shape))
    #     var = np.sum(freq * np.power(word_embs - mean, 2.), axis=0)
    #     stddev = np.sqrt(1e-6 + var)
    #     print('var:{}'.format(var.shape))
    #     print('stddev:{}'.format(stddev.shape))
    #
    #     word_embs_norm = (word_embs - mean) / stddev
    #     word_embs_norm = word_embs_norm.astype(np.float32)
    #     print('word_embs_norm:{}'.format(word_embs_norm))
    #     print(word_embs_norm)
    #     print('norm(word_embs_norm):')
    #     print(np.linalg.norm(word_embs_norm, axis=1).reshape(-1, 1))
    #     model.word_embed.W.data[:] = word_embs_norm
    #     print('#done')

    if args.word_drop:
        drop_probs = np.array([float(vocab_count.get(w, 1)) for w, idx in sorted(
            vocab.items(), key=lambda x: x[1])], dtype=np.float32)
        drop_probs = args.word_drop_prob / (args.word_drop_prob + drop_probs)
        model.drop_probs = to_gpu(drop_probs.astype(np.float32)[..., None])
        # print 'drop_probs:', drop_probs.shape
        # print drop_probs

    if args.gpu >= 0:
        model.to_gpu()
        if args.use_salience:
            trained_lstm.to_gpu()

    if all_nn_flag and args.online_nn == 0:
        model.all_nearest_words(top_k=args.nn_k)
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

        if args.dataset == 'conll_2014' or args.dataset == 'fce':

            most_sims('like')
            most_sims('likes')
            most_sims('for')
            most_sims('of')
            most_sims('to')
            # most_sims(u'preventing') # preventing
            most_sims('towards') # at
            most_sims('come') # comes
            # most_sims(u'commitment') # commitments


        else:
            most_sims(u'good')
            most_sims(u'this')
            most_sims(u'that')
            most_sims(u'awesome')
            most_sims(u'bad')
            most_sims(u'wrong')

    # if args.norm_emb:
    #     model.norm_word_emb()

    def evaluate(x_set, x_length_set, y_set):
        chainer.config.train = False
        chainer.config.enable_backprop = False
        iteration_list = range(0, len(x_set), batchsize)
        correct_cnt = 0
        correct_cnt_wrong_only = 0
        correct_cnt_sent = 0
        total_cnt = 0.0
        total_cnt_wrong_only = 0.0
        total_cnt_sent = 0.0
        avg_rate = 0.0
        avg_rate_num = 0.0
        TP = 0.0
        TP_FP = 0.0
        TP_FN = 0.0
        y_np = []
        predicted_np = []

        for i_index, index in enumerate(iteration_list):
            x = [to_gpu(_x) for _x in x_set[index:index + batchsize]]
            x_length = x_length_set[index:index + batchsize]

            if args.use_seq_labeling:
                y_flat = np.concatenate(y_set[index:index + batchsize], axis=0).astype(np.int32)
                y = to_gpu(y_flat)
            else:
                y = to_gpu(y_set[index:index + batchsize])

            salience = None
            if args.use_salience:
                chainer.config.enable_backprop = True
                chainer.config.train = True
                salience_lstm = trained_lstm
                output = salience_lstm(x, x_length)
                pred_y = F.argmax(output, axis=1)
                idx = xp.arange(pred_y.shape[0], dtype=xp.int32)
                salience_score = F.sum(output[idx, pred_y.data])
                salience_lstm.cleargrads()
                salience_score.backward()
                salience = salience_lstm.word_embed.W.grad
                salience = [chainer.Variable(salience[_x]) for _x in x]
                chainer.config.enable_backprop = False
                chainer.config.train = False

            # output = model(x, x_length, salience=salience)
            output = model(x, x_length)

            if args.use_crf:
                predicted_list, y_list_sorted = model.crf_loss(output, x_length, y, ign_loss=True)

            if args.use_crf:
                predict_flat = xp.concatenate(predicted_list, axis=0)
                y_flat = xp.concatenate(y_list_sorted, axis=0)
                correct_cnt += xp.sum(predict_flat == y_flat)
                idx = y_flat == 0
                correct_cnt_wrong_only += xp.sum(predict_flat[idx] == y_flat[idx])
                total_cnt_wrong_only += len(y_flat[idx])

                if args.use_seq_labeling:
                    pred_sent = xp.split(predict_flat, np.cumsum(x_length)[:-1])
                    y_sent = xp.split(y_flat, np.cumsum(x_length)[:-1])
                    correct_cnt_sent += sum([(p_ == y_).all() for p_, y_ in zip(pred_sent, y_sent)])

                    y_np.append(to_cpu(y_flat))
                    predicted_np.append(to_cpu(predict_flat))
                    # TP_FP += xp.sum(predict_flat == 1)
                    # TP_FN += xp.sum(y_flat == 1)
                    # flag_y = y_flat == 1
                    # flag_pred = predict_flat == 1
                    #
                    # TP += xp.sum(flag_y * flag_pred)
            else:
                predict = xp.argmax(output.data, axis=1)
                correct_cnt += xp.sum(predict == y)
                idx = y == 0
                correct_cnt_wrong_only += xp.sum(predict[idx] == y[idx])
                total_cnt_wrong_only += len(y[idx])

                if args.use_seq_labeling:
                    pred_sent = xp.split(predict, np.cumsum(x_length)[:-1])
                    y_sent = xp.split(y, np.cumsum(x_length)[:-1])
                    correct_cnt_sent += sum([(p_ == y_).all() for p_, y_ in zip(pred_sent, y_sent)])


                    # TP_FP += xp.sum(predict == 1)
                    # TP_FN += xp.sum(y == 1)
                    # flag_y = y_flat == 1
                    # flag_pred = predict_flat == 1
                    # TP += xp.sum(flag_y * flag_pred)
                    y_np.append(to_cpu(y))
                    predicted_np.append(to_cpu(predict))
            total_cnt += len(y)
            total_cnt_sent += len(x)

            # z_norm_loss = model.z_norm_loss
            # avg_rate += F.sum(z_norm_loss).data
            # avg_rate_num += z_norm_loss.shape[0]
        if args.accuracy_f1:
            # precision = TP / (TP_FP if TP_FP > 0.0 else 1.0)
            # recall = TP / (TP_FN if TP_FN > 0.0 else 1.0)
            # f1_score = (2 * recall * precision) / (recall + precision)
            y_np = np.concatenate(y_np, axis=0)
            predicted_np = np.concatenate(predicted_np, axis=0)
            # precision, recall, f1_score, support = precision_recall_fscore_support(y_np, predicted_np)
            # logging.info('        precision:{}  recall:{}  f1_score:{}'.format(precision, recall, f1_score))

            # f1_score_val = f1_score_func(y_np, predicted_np, average='binary', pos_label=0)
            # precision, recall, f1_score, support = precision_recall_fscore_support(y_np, predicted_np, average='micro')

            precision, recall, f1_score, support = precision_recall_fscore_support(y_np, predicted_np, average='binary', beta=0.5, pos_label=0)
            log_str = '       precision:{}, recall:{}, f1_score:{}'.format(str(precision), str(recall), str(f1_score))
            logging.info(log_str)
            # print 'precision, recall, f1_score, support:', precision, recall, f1_score, support
            # print 'y_np:', y_np.shape, np.sum(y_np == 1), np.sum(y_np == 0)
            # print 'predicted_np:', predicted_np.shape, np.sum(predicted_np == 1), np.sum(predicted_np == 0)
            f1_score_val = np.average(f1_score)
            accuracy = f1_score_val
        elif args.accuracy_sentence:
            accuracy = correct_cnt_sent / total_cnt_sent
        elif args.accuracy_for_wrong:
            accuracy = correct_cnt_wrong_only / total_cnt_wrong_only
        else:
            accuracy = correct_cnt / total_cnt
        chainer.config.enable_backprop = True
        # avg_rate = avg_rate / avg_rate_num
        return accuracy * 100.0, avg_rate

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
    if args.l2:
        opt.add_hook(chainer.optimizer.WeightDecay(args.l2))

    if args.freeze_word_emb:
        model.freeze_word_emb()
    if args.kmeans:
        model.get_clusters_words()

    prev_dev_accuracy = 0.0
    global_step = 0.0
    adv_rep_num_statics = {}
    adv_rep_pos_statics = {}

    if args.debug_eval:
        dev_accuracy, dev_avg_rate = evaluate(dev_x, dev_x_len, dev_y)
        log_str = ' [dev] accuracy:{}, length:{}'.format(str(dev_accuracy), str(dev_avg_rate))
        logging.info(log_str)

        # test
        test_accuracy, test_avg_rate = evaluate(test_x, test_x_len, test_y)
        log_str = ' [test] accuracy:{}, length:{}'.format(str(test_accuracy), str(test_avg_rate))
        logging.info(log_str)


    for epoch in range(args.n_epoch):
        logging.info('epoch:' + str(epoch))
        # train
        model.cleargrads()
        model.reset_statics()
        chainer.config.train = True
        iteration_list = range(0, len(train_x), batchsize)
        if args.analysis_mode:
            iteration_list = range(0, len(test_sorted), batchsize)
            chainer.config.train = False
            chainer.config.enable_backprop = True
            chainer.config.cudnn_deterministic = True
            chainer.config.use_cudnn = 'never'

        # iteration_list_semi = range(0, len(semi_train_x), batchsize)
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
        for i_index, index in enumerate(iteration_list):
            global_step += 1.0
            model.set_train(True)
            sample_idx = perm[index:index + batchsize]
            x = [to_gpu(train_x[_i]) for _i in sample_idx]
            x_length = [train_x_len[_i] for _i in sample_idx]

            if args.use_seq_labeling:
                y_flat = np.concatenate([train_y[_i] for _i in sample_idx], axis=0).astype(np.int32)
                y = to_gpu(y_flat)
            else:
                y = to_gpu(train_y[sample_idx])


            if args.analysis_mode:
                if i_index >= len(test_sorted) - 1:

                    with open(args.save_name, mode='wb') as f:
                        pickle.dump(save_items, f)
                    return False
                sample_idx = [test_sorted[i_index]]
                x = [to_gpu(test_x[_i]) for _i in sample_idx]
                x_length = [test_x_len[_i] for _i in sample_idx]

                if args.use_seq_labeling:
                    y_flat = np.concatenate([test_y[_i] for _i in sample_idx], axis=0).astype(np.int32)
                    y = to_gpu(y_flat)
                else:
                    y = to_gpu(test_y[sample_idx])
                    # y = to_gpu(test_y[sample_idx])

                if args.use_seq_labeling:
                    _correct_x = [correct_x[_i] for _i in sample_idx]
                    fil_type = args.fil_type
                    if fil_type == 0:
                        nearest_ids_grammar = xp.ones(x[0].shape[0]).astype('i')
                    else:
                        nearest_ids_grammar = - xp.ones(x[0].shape[0]).astype('i')
                        vocab_inv[-1] = '<PAD>'
                    y_incorrect_idx = to_cpu(xp.arange(y.shape[0])[y == 0])
                    nearest_ids = model.get_nearest_words(xp.concatenate(x, axis=0))

                    for _inc_idx in y_incorrect_idx:
                        s_x = x[0][int(_inc_idx)]
                        t_x = _correct_x[0][int(_inc_idx)]

                        # print 's_x:', s_x, vocab_inv[int(s_x)]
                        # print 't_x:', t_x, vocab_inv[int(t_x)]
                        if fil_type == 0:
                            nearest_ids_grammar[int(_inc_idx)] = t_x
                        else:
                            if int(t_x) not in nearest_ids[int(_inc_idx)].tolist():
                                nearest_ids_grammar[int(_inc_idx)] = t_x


                    model.nearest_ids_grammar = nearest_ids_grammar.reshape((-1, 1))

                original_words = [vocab_inv[int(idx)] for idx in test_x[sample_idx[0]]]
                logging.info('i_index:{}'.format(i_index))
                logging.info('x:{} {}'.format(x[0].shape[0], ' '.join(original_words)))
                logging.info('y:{}'.format(y))

            d = None
            d_hidden = None

            loss = 0.0
            eps_statics = 0.0
            if args.random_noise:
                x_concat = xp.concatenate(x, axis=0)
                d_rand = xp.random.normal(size=(x_concat.shape[0], args.emb_dim), dtype='f')
                output_rand = model(x, x_length, d=d_rand, first_step=False, normal_noise=True)

                loss += F.softmax_cross_entropy(output_rand, y, normalize=True)

            if args.use_adv or args.use_adv_hidden or args.use_semi_data:
                output = model(x, x_length)
                output_original = output

                if args.use_crf:
                    loss_crf, predicted_list, y_list_sorted = model.crf_loss(output, x_length, y, ign_loss=False)
                    loss += loss_crf
                else:
                    if args.sent_loss_usual:
                        loss_sent = F.softmax_cross_entropy(output, y, reduce='no')
                        loss_sent_list = F.split_axis(loss_sent, np.cumsum(x_length)[:-1], axis=0)
                        loss_sent_list = F.concat(loss_sent_list, axis=0)
                        loss += F.sum(loss_sent_list) / loss_sent_list.shape[0]
                    else:
                        loss += F.softmax_cross_entropy(output, y, normalize=True)
                init_d = None
                init_d_attn = None
                if args.double_adv:
                    output = model(x, x_length, first_step=True, normal_noise=True)
                    first_loss_adv_first = F.softmax_cross_entropy(output, y, normalize=True)
                    model.cleargrads()
                    first_loss_adv_first.backward()
                    d = model.d_var.grad
                    if args.up_grad:
                        d = model.d_var + model.d_var.grad
                    init_d = d
                    output = model(x, x_length, d=d, d_hidden=d_hidden, normal_noise=True)
                    loss_adv = F.softmax_cross_entropy(output, y, normalize=True)
                    loss += loss_adv * args.nl_factor

                for adv_i in range(args.adv_iter):
                    # logging.info('#adv_iter:{}'.format(adv_i))
                    if args.use_adv or args.use_adv_hidden:
                        output = model(x, x_length, first_step=True, d=init_d, init_d_attn=init_d_attn)

                        if args.use_crf:
                            loss_crf, predicted_list, y_list_sorted = model.crf_loss(output, x_length, y, ign_loss=False)
                            loss_adv_first = loss_crf
                        else:

                            if args.sent_loss:
                                loss_sent = F.softmax_cross_entropy(output, y, reduce='no')
                                loss_sent_list = F.split_axis(loss_sent, np.cumsum(x_length)[:-1], axis=0)
                                loss_sent_list = F.concat(loss_sent_list, axis=0)
                                loss_adv_first = F.sum(loss_sent_list) / loss_sent_list.shape[0]
                            else:
                                loss_adv_first = F.softmax_cross_entropy(output, y, normalize=True)


                            if args.analysis_mode:
                                if xp.sum(y == 0):
                                    # print 'F.softmax_cross_entropy:', F.softmax_cross_entropy(output, y, reduce='no').shape
                                    # loss_adv_first = F.softmax_cross_entropy(output, y, reduce='no')[y == 0]
                                    # print 'loss_adv_first:', loss_adv_first.shape
                                    # loss_adv_first = F.sum(loss_adv_first)
                                    if args.analysis_loss == 0:
                                        y_wrong = xp.zeros(y.shape).astype('i')
                                        loss_adv_first = F.softmax_cross_entropy(output, y_wrong, normalize=True)
                                    elif args.analysis_loss == 1:
                                        loss_adv_first = F.softmax_cross_entropy(output, y, reduce='no')[y == 0]
                                        loss_adv_first = F.sum(loss_adv_first)


                        if args.use_saliency:
                            loss_adv_first = - F.sum(output[y])

                        # if args.debug:
                        #     print 'loss_adv_first:', loss_adv_first
                        if args.use_nn_term:
                            nn_terms = model.nn_terms
                            if args.nn_term_sq:
                                if args.nn_term_sq_half:
                                    loss_adv_first += args.loss_eps * (- (0.5 - nn_terms) ** 2 + 1)
                                else:
                                    loss_adv_first += args.loss_eps * (- (1.0 - nn_terms) ** 2 + 1)
                            elif args.reverse_loss:
                                loss_adv_first += - args.loss_eps * nn_terms
                            else:
                                loss_adv_first += args.loss_eps * nn_terms
                            # if args.debug:
                            #     print 'nn_terms:', nn_terms
                            #     print '(1.0 - nn_terms) ** 2:', (1.0 - nn_terms) ** 2

                        model.cleargrads()
                        if args.init_grad:
                            model.attention_d_var.grad = xp.ones(model.attention_d_var.shape).astype(xp.float32)
                        loss_adv_first.backward()
                        attn_d_grad_bf = None
                        if args.use_adv:
                            if args.use_attn_d:
                                attn_d_grad = model.attention_d_var.grad
                                attn_d_grad_norm = xp.linalg.norm(attn_d_grad, axis=tuple(range(1, len(attn_d_grad.shape))))
                                if args.use_attn_sent_norm:
                                    # print 'attn_d_grad:', attn_d_grad.shape
                                    original_shape = attn_d_grad.shape
                                    dim = attn_d_grad.shape[1]
                                    d_list = F.split_axis(attn_d_grad, np.cumsum(x_length)[:-1], axis=0)
                                    max_length = np.max(x_length)
                                    d_pad = F.pad_sequence(d_list, length=max_length, padding=0.0)
                                    d_flat = net.get_normalized_vector(d_pad, None)
                                    d_flat = F.reshape(d_flat, (-1, dim))
                                    split_size = np.cumsum(np.full(len(x), max_length))[:-1]
                                    d_list = F.split_axis(d_flat, split_size, axis=0)
                                    d_list = [_d[:_length_tmp] for _d, _length_tmp in zip(d_list, x_length)]
                                    attn_d_grad = F.concat(d_list, axis=0)
                                    attn_d_grad = F.reshape(attn_d_grad, original_shape)
                                    # print 'attn_d_grad:', attn_d_grad.shape



                                if args.use_grad_scale or args.analysis_mode:
                                    grad_list = F.split_axis(attn_d_grad_norm, np.cumsum(x_length)[:-1], axis=0)
                                    if args.scale_type == 0:
                                        grad_list = [_grad.data / (xp.max(_grad.data, keepdims=True)+1e-12) for _grad in grad_list]
                                    elif args.scale_type == 1:
                                        grad_list = [_grad.data / (xp.sum(_grad.data, keepdims=True)+1e-12) for _grad in grad_list]
                                    elif args.scale_type == 10:
                                        def scale_func(_grad):
                                            _grad = _grad / (xp.sum(_grad, keepdims=True)+1e-12)
                                            _grad = _grad / (xp.max(_grad, keepdims=True)+1e-12)
                                            return _grad
                                        grad_list = [scale_func(_grad.data) for _grad in grad_list]
                                    elif args.scale_type == 2:
                                        grad_list = [_grad.data / (xp.linalg.norm(_grad.data)+1e-12) for _grad in grad_list]
                                    # print 'grad_list:', grad_list
                                    grad_scale = xp.concatenate(grad_list, axis=0)
                                    if len(grad_scale.shape) == 1:
                                        grad_scale = grad_scale[..., None]
                                    model.grad_scale = grad_scale
                                    # print 'attn_d_grad:', attn_d_grad.shape
                                    # print 'grad_scale:', grad_scale.shape

                                if args.fixed_d:
                                    attn_d_grad = model.xs_noise.grad
                                attn_d_grad_original = attn_d_grad
                                if args.norm_sent_attn_scala:
                                    attn_d_grad = model.norm_vec_sentence_level_func(attn_d_grad)
                                    attn_d_grad = F.reshape(attn_d_grad, (attn_d_grad.shape[0], -1, 1))

                                if args.up_grad or args.up_grad_attn:
                                    attn_d_grad = model.attention_d_var.data + model.attention_d_var.grad
                                if args.no_grad:
                                    attn_d_grad = model.attention_d_var.data
                                diff_data = model.diff
                                diff_data_here = model.diff_data_here

                                if args.freeze_nn:
                                    diff_data_here = diff_data_here.data

                                if args.use_plus_d:
                                    d_plus_grad = model.d_plus.grad
                                    if args.freeze_d_plus:
                                        d_plus_grad = model.d_plus.data

                                    # d_plus_grad /= xp.linalg.norm(d_plus_grad, axis=2, keepdims=True)
                                    if args.scala_plus:
                                        diff_data_here = F.concat([d_plus_grad, diff_data_here], axis=1).data

                                if args.use_plus_zeros:
                                    d_plus_shape = (diff_data.shape[0], 1, diff_data.shape[2])
                                    d_plus_zero = xp.zeros(d_plus_shape, dtype='f')
                                    diff_data_here = F.concat([d_plus_zero, diff_data_here], axis=1).data

                                if args.use_attn_dot:
                                    original_shape = attn_d_grad.shape
                                    attn_d_grad = F.reshape(attn_d_grad, (attn_d_grad.shape[0], -1))
                                    attn_d_grad = F.normalize(attn_d_grad, axis=1)
                                    attn_d_grad = F.reshape(attn_d_grad, original_shape)
                                    if attn_d_grad.shape != diff_data_here.shape:
                                        attn_d_grad = F.broadcast_to(attn_d_grad, diff_data_here.shape)

                                    dot = F.sum(attn_d_grad * diff_data_here, axis=2)
                                    # dot = F.batch_matmul(attn_d_grad, diff_data_here, transb=True)
                                    dot = F.reshape(dot, (dot.shape[0], dot.shape[1], 1))
                                    # print 'dot:', dot.shape, dot.data.reshape(-1)
                                    attn_d_grad = dot


                                if args.attentional_d_mode:
                                    # most

                                    d_var_norm = net.get_normalized_vector(attn_d_grad, None)
                                    d_var_norm = F.reshape(d_var_norm, (d_var_norm.shape[0], 1, -1))
                                    # attn_d_grad_norm = net.get_normalized_vector(attn_d_grad, xp)
                                    # attn_d_grad_norm = xp.reshape(attn_d_grad_norm, (attn_d_grad_norm.shape[0], 1, -1))
                                    # attn_d_grad_norm = xp.reshape(attn_d_grad_norm, (attn_d_grad_norm.shape[0], -1, -1))
                                    d_norm = net.get_normalized_vector(model.diff_data, xp, (2))
                                    # d_norm = xp.reshape(d_norm, (d_norm.shape[0], 1, -1))
                                    attn_d_grad = F.matmul(d_norm, d_var_norm, False, True)
                                    # print 'sims:', sims.shape
                                    # attn_d_grad = F.reshape(sims, (sims.shape[0], -1))
                                    # attn_d_grad = sims
                                    # print 'attn_d_grad:', attn_d_grad.shape
                                    # print 'diff_data_here:', diff_data_here.shape



                                if args.debug:
                                    attention_d_var = model.attention_d_var
                                    # print 'attention_d_var:', attention_d_var.shape, xp.linalg.norm(attention_d_var.data, axis=1).reshape(-1)
                                if args.use_softmax == 1:
                                    attn_d_grad = F.softmax(attn_d_grad * args.soft_int_final, axis=1)
                                elif args.use_softmax == -1:
                                    attn_d_grad = F.normalize(attn_d_grad, axis=1)
                                elif args.use_softmax == 2:
                                    attn_d_grad = F.absolute(attn_d_grad)
                                elif args.use_softmax == -21:
                                    attn_d_grad = F.normalize(F.absolute(attn_d_grad), axis=1)
                                elif args.use_softmax == -2:
                                    attn_d_grad = F.normalize(attn_d_grad, axis=1)
                                    term = F.reshape(F.sum(F.absolute(attn_d_grad), axis=1), (attn_d_grad.shape[0], 1, -1))
                                    term = F.broadcast_to(term, attn_d_grad.shape)
                                    attn_d_grad = attn_d_grad / (term + 1e-12)
                                elif args.use_softmax == 3:
                                    term = F.reshape(F.max(F.absolute(attn_d_grad), axis=1), (attn_d_grad.shape[0], 1, -1))
                                    term = F.broadcast_to(term, attn_d_grad.shape)
                                    attn_d_grad = attn_d_grad / (term + 1e-12)
                                elif args.use_softmax == -3:
                                    term = F.reshape(F.min(F.absolute(attn_d_grad), axis=1), (attn_d_grad.shape[0], 1, -1))
                                    term = F.broadcast_to(term, attn_d_grad.shape)
                                    attn_d_grad = attn_d_grad / (term + 1e-12)
                                elif args.use_softmax == -11:
                                    attn_d_grad = F.softmax(F.normalize(attn_d_grad, axis=1) * args.soft_int_final, axis=1)
                                elif args.use_softmax == -12:
                                    attn_d_grad = F.softmax(F.normalize(F.log(attn_d_grad), axis=1), axis=1)
                                elif args.use_softmax == -111:
                                    term = F.reshape(F.max(F.absolute(attn_d_grad), axis=1), (attn_d_grad.shape[0], 1, -1))
                                    term = F.broadcast_to(term, attn_d_grad.shape)
                                    attn_d_grad = attn_d_grad / (term + 1e-12)
                                    attn_d_grad = F.softmax(attn_d_grad, axis=1)
                                attn_d_grad_bf = attn_d_grad
                                if args.max_attn:
                                    attn_d_grad_data = attn_d_grad.data if isinstance(attn_d_grad, chainer.Variable) else attn_d_grad
                                    if args.max_attn_type == 0:
                                        max_flags = xp.max(attn_d_grad_original, axis=1, keepdims=True) == attn_d_grad_original
                                        attn_d_grad = xp.where(max_flags, 1., 0.).astype('f')
                                    if args.max_attn_type == -1:
                                        max_flags = xp.max(attn_d_grad_original, axis=1, keepdims=True) == attn_d_grad_original
                                        plus_flags = attn_d_grad_original > 0.0
                                        attn_d_grad = xp.where(max_flags * plus_flags, 1., 0.).astype('f')
                                    elif args.max_attn_type == 1:
                                        max_flags = xp.max(attn_d_grad_original, axis=1, keepdims=True) == attn_d_grad_original
                                        attn_d_grad = xp.where(max_flags, F.normalize(attn_d_grad, axis=1).data, 0.0)
                                    elif args.max_attn_type == 2:
                                        max_flags = xp.max(attn_d_grad_original, axis=1, keepdims=True) == attn_d_grad_original
                                        attn_d_grad = xp.where(max_flags, attn_d_grad_data, 0.0)
                                    elif args.max_attn_type == -2:
                                        max_flags = xp.max(attn_d_grad_original, axis=1, keepdims=True) == attn_d_grad_original
                                        plus_flags = attn_d_grad_original > 0.0
                                        attn_d_grad = xp.where(max_flags * plus_flags, attn_d_grad_data, 0.0)
                                    elif args.max_attn_type == 3:
                                        max_flags = xp.max(attn_d_grad_original, axis=1, keepdims=True) == attn_d_grad_original
                                        attn_d_grad = xp.where(max_flags, attn_d_grad_data, 0.0)
                                    elif args.max_attn_type == 4:
                                        max_flags = xp.max(attn_d_grad_original, axis=1, keepdims=True) == attn_d_grad_original
                                        attn_d_grad = xp.where(max_flags, 1., 0.).astype('f')
                                        attn_d_grad = attn_d_grad * attn_d_grad_norm[..., None]
                                    elif args.max_attn_type == -10:
                                        max_flags_idx = xp.argmax(xp.random.gumbel(size=attn_d_grad_data.shape) + attn_d_grad_data, axis=1).reshape(-1)
                                        max_flags = xp.eye(attn_d_grad_data.shape[1])[max_flags_idx].reshape((attn_d_grad_data.shape[0], attn_d_grad_data.shape[1], 1))
                                        plus_flags = attn_d_grad_data > 0.0
                                        attn_d_grad = xp.where(max_flags * plus_flags, 1., 0.).astype('f')

                                    elif args.max_attn_type == 42:
                                        tmp_attn_d = F.broadcast_to(attn_d_grad, diff_data_here.shape).data
                                        tmp_d = F.sum(tmp_attn_d * diff_data_here, axis=1)
                                        tmp_d_norm = net.get_normalized_vector(tmp_d.data, xp)
                                        tmp_d_norm = xp.reshape(tmp_d_norm, (tmp_d_norm.shape[0], 1, -1))
                                        dir_norm = model.dir_norm
                                        sims = F.matmul(dir_norm, tmp_d_norm, False, True)
                                        sims = xp.reshape(sims.data, (sims.shape[0], -1))
                                        most_similars = F.argmax(sims, axis=1).data
                                        max_flags = xp.eye(attn_d_grad_data.shape[1])[most_similars].reshape((attn_d_grad_data.shape[0], attn_d_grad_data.shape[1], 1))
                                        plus_flags = attn_d_grad_data > 0.0
                                        attn_d_grad = xp.where(max_flags * plus_flags, 1., 0.).astype('f')


                                    elif args.max_attn_type == 64:
                                        tmp_attn_d = F.broadcast_to(attn_d_grad, diff_data_here.shape).data
                                        tmp_d = F.sum(tmp_attn_d * diff_data_here, axis=1)
                                        tmp_d_norm = net.get_normalized_vector(tmp_d.data, xp)
                                        tmp_d_norm_reshape = xp.reshape(tmp_d_norm, (tmp_d_norm.shape[0], 1, -1))
                                        dir_norm = model.dir_norm
                                        sims = F.matmul(dir_norm, tmp_d_norm_reshape, False, True)
                                        sims = xp.reshape(sims.data, (sims.shape[0], -1))
                                        most_similars = F.argmax(sims, axis=1).data
                                        max_flags = xp.eye(attn_d_grad_data.shape[1])[most_similars].reshape((attn_d_grad_data.shape[0], attn_d_grad_data.shape[1], 1))
                                        plus_flags = attn_d_grad_data > 0.0
                                        attn_d_grad = xp.where(max_flags * plus_flags, 1., 0.).astype('f')
                                    elif args.max_attn_type == -64:
                                        tmp_attn_d = F.broadcast_to(attn_d_grad, diff_data_here.shape).data
                                        tmp_d = F.sum(tmp_attn_d * diff_data_here, axis=1)
                                        tmp_d_norm = net.get_normalized_vector(tmp_d.data, xp)
                                        max_flags = xp.max(attn_d_grad_original, axis=1, keepdims=True) == attn_d_grad_original
                                        plus_flags = attn_d_grad_original > 0.0
                                        attn_d_grad = xp.where(max_flags * plus_flags, 1., 0.).astype('f')
                                    elif args.max_attn_type == -100:
                                        attn_d_grad_filter = xp.where(attn_d_grad_data > 0.0, attn_d_grad_data, 0.0)
                                        tmp_attn_d = F.broadcast_to(attn_d_grad_filter, diff_data_here.shape).data
                                        tmp_d = F.sum(tmp_attn_d * diff_data_here, axis=1)
                                        tmp_d_norm = net.get_normalized_vector(tmp_d.data, xp)
                                        max_flags = xp.max(attn_d_grad_original, axis=1, keepdims=True) == attn_d_grad_original
                                        plus_flags = attn_d_grad_original > 0.0
                                        attn_d_grad = xp.where(max_flags * plus_flags, 1., 0.).astype('f')
                                    if args.use_random_max:
                                        rnd_idx = xp.random.randint(attn_d_grad_data.shape[1], size=(attn_d_grad_data.shape[0], ))
                                        max_flags = xp.eye(attn_d_grad_data.shape[1])[rnd_idx].reshape((attn_d_grad_data.shape[0], attn_d_grad_data.shape[1], 1))
                                        attn_d_grad = xp.where(max_flags, 1., 0.).astype('f')

                                    # TODO: Plusの部分だけ有効にする
                                    #


                                # print 'attn_d_grad:', attn_d_grad.shape
                                # print 'diff_data_here:', diff_data_here.shape
                                attn_d = F.broadcast_to(attn_d_grad, diff_data_here.shape).data
                                d = F.sum(attn_d * diff_data_here, axis=1)
                                if args.div_attn_d:
                                    d = d / diff_data_here.shape[1]

                                if args.use_plus_d:
                                    if not args.scala_plus:
                                        d += F.sum(d_plus_grad, axis=1)

                                if args.max_attn_type == 64 or args.max_attn_type == -64 or args.max_attn_type == -100:
                                    d = args.noise_factor * d +  tmp_d_norm * (1. - args.noise_factor)

                                init_d_attn = attn_d_grad.data if isinstance(attn_d_grad, chainer.Variable) else attn_d_grad
                            else:
                                d = model.d_var.grad
                                if args.fixed_d:
                                    d = model.xs_noise.grad


                                attn_d_grad = chainer.Variable(d)
                                attn_d_grad_original = d
                                if args.up_grad:
                                    d = model.d_var.data + model.d_var.grad

                            d_data = d.data if isinstance(d, chainer.Variable) else d
                            init_d = d_data

                        if args.use_adv_hidden:
                            d_hidden = model.d_var_hidden.grad
                        output = model(x, x_length, d=d, d_hidden=d_hidden)

                        if args.use_crf:
                            loss_crf, predicted_list, y_list_sorted = model.crf_loss(output, x_length, y, ign_loss=False)
                            loss_adv = loss_crf
                        else:
                            loss_adv = F.softmax_cross_entropy(output, y, normalize=True)
                        loss += loss_adv * args.nl_factor

                        if args.adv_mode:
                            if i_index > args.adv_mode_iter and args.analysis_mode == 0 or i_index == N -1:
                                with open(args.save_name + '_craft.pickle', mode='wb') as f:
                                    pickle.dump([adv_rep_num_statics, adv_rep_pos_statics], f)
                                return False
                            predict_adv = xp.argmax(output.data, axis=1)
                            # print 'output_original:', output_original.shape
                            predict = xp.argmax(output_original.data, axis=1)
                            exact_same_result = (predict == y).all()
                            logging.info('predict:y:{}, {}, {}'.format(exact_same_result, predict, y))
                            # print 'predict:y', predict.shape, y.shape
                            # print predict != y

                            if args.use_seq_labeling:
                                wrong_idx = y == 0
                                # print 'predict[wrong_idx]:', predict[wrong_idx]
                                # print 'y[wrong_idx]:', y[wrong_idx]

                                diff = predict[wrong_idx] != y[wrong_idx]

                                flag = diff.any() or diff.shape[0] == 0
                                # print 'diff:', diff.shape, diff
                                # print 'flag:', flag
                            else:
                                flag = predict != y
                            if flag:
                                # print '[continue] predict:y', predict, y
                                continue
                            adv_rep_pos_statics[i_index] = []

                            x_concat = xp.concatenate(x, axis=0)
                            # TODO: memo : comment out top_idx
                            '''
                            top_idx = model.get_nearest_words(x_concat, noise=model.xs_noise_final_data, ign_offset=True)
                            rep_ids = top_idx[:, 0]
                            '''
                            rep_ids = x_concat
                            if args.print_info:
                                # print 'attn_d_grad:', attn_d_grad.shape, attn_d_grad
                                # if model.grad_scale is not None:
                                #     print 'model.grad_scale:', model.grad_scale.shape, model.grad_scale
                                if attn_d_grad_bf is not None:
                                    # print 'attn_d_grad_bf:', attn_d_grad_bf.shape, attn_d_grad_bf
                                    # logging.info('    rate: {}   ({}) \t disc:{} \tcraft_disc:{} \t is_adv:{}, is_adv_craft:{}'.format(rate, len(is_adv_example_list), disc_rate, disc_craft_rate, is_adv_example, succes_craft_adv_example))
                                    attn_d_grad_bf_data = attn_d_grad_bf.data if isinstance(attn_d_grad_bf, chainer.Variable) else attn_d_grad_bf
                                    # print 'max(attn_d_grad_bf):', xp.max(attn_d_grad_bf_data, axis=1)

                                if attn_d_grad_original is not None:
                                    attn_d_grad_original_data = attn_d_grad_original.data if isinstance(attn_d_grad_original, chainer.Variable) else attn_d_grad_original
                                    # print 'attn_d_grad_original_data:', attn_d_grad_original_data.shape, attn_d_grad_original_data

                            # print 'x_concat:', x_concat.shape
                            # print 'rep_ids:', rep_ids.shape
                            output_discrete = model([rep_ids], x_length)
                            predict_discrete_adv = xp.argmax(output_discrete.data, axis=1)
                            # print 'predict_discrete_adv:', predict_discrete_adv

                            diff_cnt = xp.sum(rep_ids != x_concat)
                            original_words = [vocab_inv[int(idx)] for idx in x_concat]
                            words = [vocab_inv[int(_rep_idx)] for _rep_idx in rep_ids]
                            rep_info = ['{} => {}'.format(vocab_inv[int(_o_idx)], vocab_inv[int(_rep_idx)]) for (_o_idx, _rep_idx) in zip(x_concat, rep_ids) if _o_idx != _rep_idx]
                            is_adv_example = predict_adv != predict
                            is_adv_example_discrete = predict_discrete_adv != predict

                            if args.use_seq_labeling:
                                is_adv_example = is_adv_example.any()
                                is_adv_example_discrete = is_adv_example_discrete.any()

                            attn_d_grad_data = attn_d_grad.data if isinstance(attn_d_grad, chainer.Variable) else attn_d_grad
                            attn_d_grad = attn_d_grad_data.reshape((attn_d_grad.shape[0], -1))
                            d_data = d.data if isinstance(d, chainer.Variable) else d
                            if args.ign_noise_eos:
                                attn_d_grad[-1, :] = 0.
                                # d_data[-1, :] = 0.
                                attn_d_grad_original[-1, :] = 0.
                            attn_d_grad_abs = F.absolute(attn_d_grad).data
                            noise_d_norm_original = xp.linalg.norm(attn_d_grad_original, axis=1).reshape(-1)
                            noise_d_norm = xp.linalg.norm(d_data, axis=1).reshape(-1)

                            # print 'attn_d_grad:', attn_d_grad.shape, attn_d_grad

                            max_nn_idx = xp.max(attn_d_grad, axis=1)
                            # print 'max_nn_idx:', max_nn_idx.shape, max_nn_idx
                            # sentiment_word_pairs = [(u'good', u'bad'), (u'right', u'wrong'), (u'happy', u'unhappy'), (u'credible', u'incredible'), (u'awesome', u'awful'), (u'great', u'little')]

                            # print 'noise_d_norm:', noise_d_norm.shape, noise_d_norm
                            # print xp.sum(noise_d_norm) / noise_d_norm.shape[0]

                            diff_data_here = model.diff_data_here.data
                            diff_norm = xp.linalg.norm(diff_data_here, axis=2).reshape(-1)
                            xi_vars = model.xi_vars
                            norm_diff_mostsim = model.norm_diff_mostsim
                            # print('model.xs_noise_final_data:', model.xs_noise_final_data.shape, xp.linalg.norm(model.xs_noise_final_data, axis=1))
                            '''
                            print 'attn_d_grad:', attn_d_grad.shape, attn_d_grad
                            print 'noise_d_norm:', noise_d_norm.shape, noise_d_norm
                            print 'diff_norm:', diff_data_here.shape, diff_norm
                            print 'xi_vars:', xi_vars.shape, xi_vars.data.reshape(-1)
                            print 'norm_diff_mostsim:',norm_diff_mostsim.shape, norm_diff_mostsim.reshape(-1)
                            '''
                            most_large_scala = xp.max(attn_d_grad, axis=1)
                            most_large_scala_idx = xp.argmax(attn_d_grad, axis=1)
                            most_large_scala_abs_idx = xp.argmax(attn_d_grad_abs, axis=1)
                            idx = xp.arange(attn_d_grad.shape[0]).astype(xp.int32)
                            most_large_scala_abs = attn_d_grad[idx, most_large_scala_abs_idx]
                            most_similars = model.most_similars
                            most_sim = model.most_sim
                            sims = model.sims
                            norm_diff_mostsim = model.norm_diff_mostsim
                            # print 'noise_d_norm:', noise_d_norm.shape, noise_d_norm.reshape(-1)
                            # print 'norm_diff_mostsim:', norm_diff_mostsim.shape, norm_diff_mostsim.reshape(-1)
                            idx = idx_func(sims.shape[0])
                            # print 'xp.argsort(-sims, axis=1):', xp.argsort(-sims, axis=1).shape
                            limit = 3
                            if args.nn_k < limit:
                                limit = args.nn_k

                            nearest_ids = model.nearest_ids_local
                            top5_sims_idx = xp.argsort(-sims, axis=1)[idx, :limit] # (batchsize, 5)
                            # top5_sims_idx_rep = [','.join([vocab_inv[int(_i)] for _i in _i_arry]) for _i_arry in top5_sims_idx]
                            # print 'top5_sims_idx:', top5_sims_idx.shape, top5_sims_idx
                            idx = idx_func(sims.shape[0])
                            idx = xp.repeat(idx, limit)
                            sorted_sims = sims[idx, top5_sims_idx.reshape(-1)].reshape(-1, limit)
                            # print 'sorted_sims:', sorted_sims.shape, sorted_sims
                            # print 'sims:', sims.shape, sims

                            most_norm_rate = norm_diff_mostsim / noise_d_norm

                            most_norm_original_idx = xp.argsort(-noise_d_norm_original, axis=0)
                            most_norm_original_idx_rev = xp.argsort(noise_d_norm_original, axis=0)
                            most_norm_diff_eps_idx = xp.argsort(-norm_diff_mostsim, axis=0)
                            most_norm_diff_eps_idx_rev = xp.argsort(norm_diff_mostsim, axis=0)
                            most_norm_rate_idx = xp.argsort(-most_norm_rate, axis=0)
                            most_norm_rate_idx_rev = xp.argsort(most_norm_rate, axis=0)
                            most_norm_idx = xp.argsort(-noise_d_norm, axis=0)
                            most_norm_idx_rev = xp.argsort(noise_d_norm, axis=0)
                            most_sims_idx_per_word = xp.argsort(-sims, axis=1)[idx, :limit]
                            most_sims_idx_top = xp.argsort(-sims, axis=1)[idx_func(sims.shape[0]), 0].reshape(-1)
                            most_sims_idx = xp.argsort(-sims[idx_func(sims.shape[0]), most_sims_idx_top].reshape(-1), axis=0)
                            most_sims_idx_rev = xp.argsort(sims[idx_func(sims.shape[0]), most_sims_idx_top].reshape(-1), axis=0)
                            most_sims_idx_values = sims[most_sims_idx, most_sims_idx_top[most_sims_idx]]
                            # most_double_idx = xp.argsort(-noise_d_norm_original * sims[idx_func(sims.shape[0]), most_sims_idx_top].reshape(-1), axis=0)
                            '''
                            print 'most_sims_idx_top:', most_sims_idx_top
                            print 'most_norm_idx:', most_norm_idx.shape, most_norm_idx.reshape(-1)
                            print 'most_sims_idx:', most_sims_idx.shape, most_sims_idx.reshape(-1)
                            print 'most_sims_idx_per_word:', most_sims_idx_per_word.shape, most_sims_idx_per_word.reshape(-1)
                            print 'most_sims_idx_values:', most_sims_idx_values.shape, most_sims_idx_values
                            '''
                            x_reps = xp.concatenate(x, axis=0)
                            # print 'x_reps:', x_reps
                            succes_craft_adv_example = False
                            log_lists = []
                            vis_lists = []
                            # for r_i in range(args.search_iters):
                            r_len = x[0].shape[0]
                            if args.use_seq_labeling:
                                r_len = sum(y == 0)

                                # print 'r_len:', r_len
                                y_incorrect_idx = to_cpu(xp.arange(y.shape[0])[y == 0])
                                # print 'y_incorrect_idx:', y_incorrect_idx
                                # print 'x:', x[0].shape, x
                                # print '_correct_x:', _correct_x[0].shape, _correct_x
                                for _inc_idx in y_incorrect_idx:
                                    # print '_inc_idx:', _inc_idx
                                    s_x = x[0][int(_inc_idx)]
                                    t_x = _correct_x[0][int(_inc_idx)]
                                    # print 's_x:', s_x, vocab_inv[int(s_x)]
                                    # print 't_x:', t_x, vocab_inv[int(t_x)]

                                if args.use_seq_labeling_pickle:
                                    r_len = x[0].shape[0]
                            is_success_grammar_flag = False
                            is_success_grammar_flag_all = True
                            r_len = min(args.search_iters, x[0].shape[0])
                            for r_i in range(r_len):
                                # idx = most_sims_idx[r_i] # word position with most sims

                                adv_type = args.adv_type
                                if adv_type == 0:
                                    if most_norm_original_idx.shape[0] - 1 < r_i:
                                        continue
                                    idx = most_norm_original_idx[r_i]
                                elif adv_type == 1:
                                    idx = most_norm_idx[r_i]
                                elif adv_type == 2:
                                    idx = most_sims_idx[r_i]
                                elif adv_type == 3:
                                    idx = most_double_idx[r_i]
                                elif adv_type == -1:
                                    idx = most_norm_original_idx_rev[r_i]
                                elif adv_type == -2:
                                    idx = most_norm_idx_rev[r_i]
                                elif adv_type == -3:
                                    idx = most_sims_idx_rev[r_i]
                                elif adv_type == 4:
                                    idx = most_norm_diff_eps_idx[r_i]
                                elif adv_type == -4:
                                    idx = most_norm_diff_eps_idx_rev[r_i]
                                elif adv_type == 5:
                                    idx = most_norm_rate_idx[r_i]
                                elif adv_type == -5:
                                    idx = most_norm_rate_idx_rev[r_i]


                                if args.analysis_mode and args.analysis_mode_type == 0:
                                    idx = r_i

                                    if args.use_seq_labeling_pickle:
                                        idx = r_i
                                    elif args.use_seq_labeling:
                                        idx = y_incorrect_idx[r_i]



                                if args.ign_noise_eos:
                                    if idx == x_reps.shape[0] - 1:
                                        continue


                                diff_idx = most_sims_idx_top[idx]
                                # TODO: most large scalerも試す

                                # max_flags = xp.max(attn_d_grad_original, axis=1, keepdims=True) == attn_d_grad_original
                                # plus_flags = attn_d_grad_original > 0.0
                                # attn_d_grad = xp.where(max_flags * plus_flags, 1., 0.).astype('f')

                                # print 'attn_d_grad_original:', attn_d_grad_original.shape
                                # print 'attn_d_grad_bf:', attn_d_grad_bf.shape
                                attn_d_grad_bf_data = attn_d_grad_bf.data if isinstance(attn_d_grad_bf, chainer.Variable) else attn_d_grad_bf
                                # print 'max(attn_d_grad_bf):', xp.max(attn_d_grad_bf_data, axis=1).shape


                                # for l_i in range(limit):
                                l_i = 0
                                '''
                                top_sim_noise_word, noise_sims = model.get_noise_sim_words(x_concat[idx].reshape(-1), d_data[idx].reshape(1, -1))
                                top_sim_noise_word = top_sim_noise_word.reshape(-1)
                                all_sim_words =  [vocab_inv[int(n_i)] for n_i in top_sim_noise_word]
                                '''
                                # print 'top_sim_noise_word:', all_sim_words
                                # print 'noise_sims:', noise_sims

                                # diff_idx = top5_sims_idx[idx, l_i]
                                logging.info('idx:{}, diff_idx:{}'.format(idx, diff_idx))
                                rep_word_idx = nearest_ids[idx, diff_idx]
                                if args.rep_sim_noise_word:
                                    rep_word_idx = top_sim_noise_word[0]

                                nn_words_list = [vocab_inv[int(n_i)] for n_i in nearest_ids[idx]]
                                nn_words = ','.join(nn_words_list)
                                top5_sims_idx_rep_here = ', '.join([vocab_inv[int(nearest_ids[idx, top5_sims_idx[idx, _i]])] for _i in  range(limit)])

                                if args.analysis_mode and args.use_seq_labeling:

                                    s_x = x[0][int(idx)]
                                    t_x = _correct_x[0][int(idx)]
                                    s_w = vocab_inv[int(s_x)]
                                    t_w = vocab_inv[int(t_x)]
                                    logging.info('Grammatical [Correct]:  {} => {}'.format(s_w, t_w))
                                    # print 't_w in nn_words_list :', t_w in nn_words_list
                                    logging.info('Grammatical [Replace]: {} => {}'.format(vocab_inv[int(x_concat[idx])], vocab_inv[int(rep_word_idx)]))
                                    is_success_grammar = t_w == vocab_inv[int(rep_word_idx)]
                                    logging.info('is_success_grammar:{}'.format(is_success_grammar))
                                    if is_success_grammar:
                                        is_success_grammar_flag = True
                                    else:
                                        is_success_grammar_flag_all = False


                                sims_here = sims[idx, diff_idx]
                                norm_here = noise_d_norm[idx]
                                norm_orig_here = noise_d_norm_original[idx]

                                attn_orig = attn_d_grad_original[idx].reshape(-1)
                                gra_sc = xp.zeros(1)
                                if attn_d_grad_bf is None:
                                    # attn_d_grad_bf = xp.zeros(attn_orig.shape).astype(xp.float32)[idx].reshape(-1)
                                    # max_scalar = 1.

                                    max_scalar = xp.max(sims, axis=1)[idx].reshape(-1)
                                    attn_d_bf = d_data[idx].reshape(-1)
                                    # gra_sc = xp.linalg.norm(d_data[idx]) / xp.linalg.norm(d_data)
                                    gra_sc = xp.linalg.norm(d_data[idx]) / xp.max(xp.linalg.norm(d_data, axis=1))
                                    # attn_d_grad_bf_data = xp.zeros(attn_d_grad_original.shape).astype(xp.float32)
                                else:
                                    attn_d_bf = attn_d_grad_bf[idx].data.reshape(-1)
                                    max_scalar = xp.max(attn_d_grad_bf_data, axis=1)[idx].reshape(-1)

                                if model.grad_scale is not None:
                                    gra_sc = model.grad_scale[idx]
                                sims_nn = sims[idx]
                                attn_d_bf = attn_d_bf[0]
                                # print 'sims_nn:', sims_nn.shape, sims_nn
                                # str_adv = ' r_i:{} \t sims:{} idx:{} diff_idx:{} norm:{} top_sims:{} {} norm_orig_here: {}\t {} => {}'.format(r_i, sims_here, idx, diff_idx, norm_here, sorted_sims[idx], top5_sims_idx_rep_here, norm_orig_here, vocab_inv[int(x_concat[idx])], vocab_inv[int(rep_word_idx)])
                                str_adv = ' {}  idx:{} {} => {}\t {} {} {} {}'.format(r_i, idx,vocab_inv[int(x_concat[idx])], vocab_inv[int(rep_word_idx)], max_scalar, attn_d_bf, nn_words, gra_sc)

                                adv_rep_pos_statics[i_index].append([vocab_inv[int(x_concat[idx])], vocab_inv[int(rep_word_idx)]])

                                log_lists.append(str_adv)
                                if args.analysis_mode and args.analysis_mode_type == 0:
                                    x_reps = xp.concatenate(x, axis=0)
                                x_reps[idx] = rep_word_idx

                                output_discrete = model([x_reps], x_length)
                                predict_discrete_adv = xp.argmax(output_discrete.data, axis=1)
                                # print ' predict_discrete_adv:', predict_discrete_adv


                                if args.use_seq_labeling:
                                    is_adv_example_discrete_craft = predict_discrete_adv != predict
                                    is_adv_example_discrete_craft = is_adv_example_discrete_craft.any()
                                else:
                                    is_adv_example_discrete_craft = predict_discrete_adv != predict
                                adv_r =  '{} is_adv_example_discrete_craft:{}'.format(r_i, is_adv_example_discrete_craft)

                                flag = is_adv_example_discrete_craft
                                flag = to_cpu(flag)
                                if args.use_seq_labeling:
                                    flag = is_success_grammar_flag

                                vis_item = [r_i, vocab_inv[int(x_concat[idx])], vocab_inv[int(rep_word_idx)], to_cpu(max_scalar), to_cpu(attn_d_bf), nn_words, to_cpu(gra_sc), flag, to_cpu(sims_nn)]
                                # print [type(_x) for _x in vis_item]
                                if args.tsne_mode:
                                    x_vec = model.word_embed(x_concat).data[idx]
                                    nn_vecs = model.word_embed(nearest_ids[idx]).data
                                    vis_item.append([to_cpu(x_vec), to_cpu(nn_vecs), to_cpu(d_data[idx]), to_cpu(xi_vars.data[idx]), to_cpu(attn_d_grad_data[idx])])
                                if args.bar_mode:
                                    # bar plot data
                                    x_vec = model.word_embed(x_concat).data[idx]
                                    nn_vecs = model.word_embed(nearest_ids[idx]).data
                                    diff_vec = nn_vecs[diff_idx].reshape(-1) - x_vec.reshape(-1)
                                    final_d_norm = xp.linalg.norm(model.xs_noise_final_data, axis=1).reshape(-1)[idx]
                                    # max_grad_float = to_cpu(xp.max(xp.linalg.norm(attn_d_grad_original, axis=1)))
                                    vis_item.append([to_cpu(diff_vec), to_cpu(final_d_norm)])
                                    logging.info('vis_item:' + str(len(vis_item)))
                                    logging.info('final_d_norm:' + str(to_cpu(final_d_norm)))


                                vis_lists.append(vis_item)
                                # logging.info(adv_r)
                                log_lists.append(adv_r)

                                if args.bar_mode:
                                    if is_adv_example_discrete_craft:
                                        succes_craft_adv_example = True
                                        log_lists.append('Number of Replaces :{}'.format(r_i+1))
                                        adv_rep_num_statics[i_index] = r_i+1
                                else:
                                    if is_adv_example_discrete_craft:
                                        succes_craft_adv_example = True
                                        log_lists.append('Number of Replaces :{}'.format(r_i+1))
                                        adv_rep_num_statics[i_index] = r_i+1
                                        break


                            if succes_craft_adv_example or is_success_grammar_flag:
                                save_item =  [vis_lists, to_cpu(x[0]), to_cpu(y)]
                                save_items.append(save_item)
                                logging.info('is_success_grammar_flag_all:{} , cnt:{}'.format(is_success_grammar_flag_all, int(xp.sum(y == 0))))
                                for log_str in log_lists:
                                    logging.info(log_str)

                            '''
                            print 'most_similars:', most_similars.shape, most_similars.reshape(-1)
                            print 'most_large_scala_idx:', most_large_scala_idx.shape, most_large_scala_idx.reshape(-1)
                            print 'most_large_scala_abs_idx:', most_large_scala_abs_idx.shape, most_large_scala_abs_idx.reshape(-1)
                            print 'most_sim:', most_sim.shape, most_sim.reshape(-1)
                            print 'most_large_scala:', most_large_scala.shape, most_large_scala.reshape(-1)
                            print 'most_large_scala_abs:', most_large_scala_abs.shape, most_large_scala_abs.reshape(-1)
                            '''
                            # TODO: crafting Disrete Adv sentence
                            ## 1. Find most Norm Noise Words
                            ## 2. Replace most Noise Words

                            ## 1. Find Most Nearest Noise Words in all words
                            ## 2. Try to check

                            ## Measures:
                            ## Most Large Norm Noise or Scalar
                            ## Most Similar Counts (sim > 0.1)
                            ##

                            # TODO: Visualize diff vectors  (good - bad)


                            # TODO:
                            # - Most Similar Diff Norm / Noise Norm
                            # Diffのノルムとノイズのノルムを使う
                            #

                            most_large_norm_words = xp.argmax(noise_d_norm.reshape(-1))
                            # print 'most_large_norm_words:', most_large_norm_words

                            # statics = [(bool(most_similars[c] == most_large_scala_idx[c]), float(most_sim[c]))  for c in range(x_concat.shape[0])]
                            statics = [(int(most_similars[c]), int(most_large_scala_idx[c]), int(most_large_scala_abs_idx[c]), float(most_sim[c]), float(noise_d_norm[c]), sorted_sims[c])  for c in range(x_concat.shape[0])]
                            # print 'statics:', statics
                            '''
                            print 'statics:'
                            for s_i in range(len(statics)):
                                print '\t', s_i, '\t', statics[s_i], top5_sims_idx[s_i], original_words[s_i], words[s_i]
                            print '\t\tis_adv_example:', is_adv_example, '\tdiff_cnt:', diff_cnt, '\tdisc:', is_adv_example_discrete, '\tcraft_adv:', succes_craft_adv_example
                            '''
                            is_adv_example_list.append(is_adv_example)
                            is_adv_example_disc_list.append(is_adv_example_discrete)
                            is_adv_example_disc_craft_list.append(succes_craft_adv_example)

                            eps_statics, sim_statics, sim_most_statics, auto_scale_statics, d_original_norm, d_norm, impscore, norm_diff, norm_diff_min, norm_noise = model.get_statics()
                            logging.info(' [train] eps:{}, sim:{}, most_sim:{}, auto_scale:{}'.format(
                                eps_statics, sim_statics, sim_most_statics, auto_scale_statics))
                            logging.info(' [train] y: {}, diff_cnt: {}, cnt_rate:({}){} n(d_ori):{}, n(d):{}, impscore:{}, n(diff):{}, n(diff_min):{}, norm_noise:{}'.format(
                                int(y[0]), diff_cnt, float(diff_cnt) / rep_ids.shape[0], rep_ids.shape[0], d_original_norm, d_norm, impscore, norm_diff, norm_diff_min, norm_noise))
                            rate = sum(is_adv_example_list) / float(len(is_adv_example_list))
                            disc_rate = sum(is_adv_example_disc_list) / float(len(is_adv_example_disc_list))
                            disc_craft_rate = sum(is_adv_example_disc_craft_list) / float(len(is_adv_example_disc_craft_list))

                            logging.info('    rate: {}   ({}) \t disc:{} \tcraft_disc:{} ({}) \t is_adv:{}, is_adv_craft:{}'.format(rate, len(is_adv_example_list), disc_rate, disc_craft_rate, len(is_adv_example_disc_craft_list), is_adv_example, succes_craft_adv_example))

                            # 各単語について載せる
                            if args.analysis_mode:
                                logging.info('******************************************')
                            #     attn_d_grad_bf_data = attn_d_grad_bf.data if isinstance(attn_d_grad_bf, chainer.Variable) else attn_d_grad_bf
                            #     for r_i in range(x[0].shape[0]):
                            #         print 'r_i:', r_i
                            #         rep_word_idx =
                            #
                            #         str_adv = ' [{}]\t{} => {} scores:{} grad_scale:{}'.format(r_i, vocab_inv[int(x_concat[r_i])], vocab_inv[int(rep_word_idx)])
                            #         # sims_here, idx, diff_idx, norm_here, sorted_sims[idx], top5_sims_idx_rep_here, norm_orig_here,
                            #
                            #
                            #
                            #




                            # if is_adv_example_discrete:
                            #     print 'rep_info:', rep_info

                d_vat = None

                if args.use_semi_data:
                    vat_iter = args.vat_iter
                else:
                    vat_iter = 0
                for vat_i in range(vat_iter):
                    if args.use_semi_data:
                        _x, _length = get_unlabled(perm_semi, i_index)
                        # print 'semi: x, _x:', len(x), len(_x)
                        output_original = model(_x, _length)

                        if args.random_noise_vat:
                            x_concat = xp.concatenate(_x, axis=0)
                            # print 'x_concat;', x_concat.shape
                            d_rand = xp.random.normal(size=(x_concat.shape[0], args.emb_dim), dtype='f')
                            # print 'd_rand:', d_rand.shape
                            output_rand = model(_x, _length, d=d_rand, first_step=False, normal_noise=True)
                            loss += net.kl_loss(xp, output_original.data, output_rand)
                            continue

                        output_vat = model(_x, _length, first_step=True, d=d_vat)

                        if args.use_semi_vat:
                            loss_vat_first = net.kl_loss(xp, output_original.data, output_vat)
                            # if args.debug:
                            #     print 'loss_vat_first:', loss_vat_first
                            # loss_vat_first = net.kl(output_original.data, output_vat)
                            # print 'loss_vat_first:', loss_vat_first
                            if args.use_nn_term:
                                nn_terms = model.nn_terms
                                # print 'nn_terms:', nn_terms
                                if args.nn_term_sq:
                                    loss_vat_first += - args.loss_eps * (1.0 - nn_terms) ** 2
                                elif args.reverse_loss:
                                    loss_vat_first += -  args.loss_eps * nn_terms
                                else:
                                    loss_vat_first += args.loss_eps * nn_terms
                            model.cleargrads()
                            # if args.debug:
                            #     print 'nn_terms:', nn_terms
                            loss_vat_first.backward()
                            if args.use_attn_d:
                                attn_d_grad = model.attention_d_var.grad
                                attn_d_grad_original = attn_d_grad
                                attn_d_grad_norm = xp.linalg.norm(attn_d_grad, axis=tuple(range(1, len(attn_d_grad.shape))))

                                if args.use_attn_sent_norm:
                                    # print 'attn_d_grad:', attn_d_grad.shape
                                    original_shape = attn_d_grad.shape
                                    dim = attn_d_grad.shape[1]
                                    d_list = F.split_axis(attn_d_grad, np.cumsum(_length)[:-1], axis=0)
                                    max_length = np.max(_length)
                                    d_pad = F.pad_sequence(d_list, length=max_length, padding=0.0)
                                    d_flat = net.get_normalized_vector(d_pad, None)
                                    d_flat = F.reshape(d_flat, (-1, dim))
                                    split_size = np.cumsum(np.full(len(_x), max_length))[:-1]
                                    d_list = F.split_axis(d_flat, split_size, axis=0)
                                    d_list = [_d[:_length_tmp] for _d, _length_tmp in zip(d_list, _length)]
                                    attn_d_grad = F.concat(d_list, axis=0)
                                    attn_d_grad = F.reshape(attn_d_grad, original_shape)
                                    # print 'attn_d_grad:', attn_d_grad.shape


                                if args.use_grad_scale:
                                    grad_list = F.split_axis(attn_d_grad_norm, np.cumsum(x_length)[:-1], axis=0)
                                    if args.scale_type == 0:
                                        grad_list = [_grad.data / (xp.max(_grad.data, keepdims=True)+1e-12) for _grad in grad_list]
                                    elif args.scale_type == 1:
                                        grad_list = [_grad.data / (xp.sum(_grad.data, keepdims=True)+1e-12) for _grad in grad_list]
                                    elif args.scale_type == 10:
                                        def scale_func(_grad):
                                            _grad = _grad / (xp.sum(_grad, keepdims=True)+1e-12)
                                            _grad = _grad / (xp.max(_grad, keepdims=True)+1e-12)
                                            return _grad

                                        grad_list = [scale_func(_grad.data) for _grad in grad_list]
                                    elif args.scale_type == 2:
                                        grad_list = [_grad.data / (xp.linalg.norm(_grad.data)+1e-12) for _grad in grad_list]
                                    # print 'grad_list:', grad_list
                                    grad_scale = xp.concatenate(grad_list, axis=0)
                                    if len(grad_scale.shape) == 1:
                                        grad_scale = grad_scale[..., None]
                                    model.grad_scale = grad_scale

                                diff_data = model.diff_data

                                diff_data_here = diff_data
                                if args.norm_diff_sent:
                                    diff_data_flat = xp.reshape(diff_data, (diff_data.shape[0] * diff_data.shape[1], -1))
                                    diff_data_here = model.norm_vec_sentence_level_func(diff_data_flat, nn_flag=True)
                                    diff_data_here = xp.reshape(diff_data_here, diff_data.shape)
                                elif args.norm_diff:
                                    diff_data_here = model.dir_norm.data

                                if args.use_plus_d:
                                    d_plus_grad = model.d_plus.grad
                                    # diff_data_here = F.concat([d_plus_grad, diff_data_here], axis=1).data

                                if args.use_softmax == 1:
                                    attn_d_grad = F.softmax(attn_d_grad * args.soft_int_final, axis=1)
                                elif args.use_softmax == -1:
                                    attn_d_grad = F.normalize(attn_d_grad, axis=1)
                                elif args.use_softmax == 2:
                                    attn_d_grad = F.absolute(attn_d_grad)
                                elif args.use_softmax == -21:
                                    attn_d_grad = F.normalize(F.absolute(attn_d_grad), axis=1)
                                elif args.use_softmax == 2:
                                    attn_d_grad = F.absolute(attn_d_grad)
                                elif args.use_softmax == -2:
                                    term = F.reshape(F.sum(F.absolute(attn_d_grad), axis=1), (attn_d_grad.shape[0], 1, -1))
                                    term = F.broadcast_to(term, attn_d_grad.shape)
                                    attn_d_grad = attn_d_grad / (term + 1e-12)
                                elif args.use_softmax == 3:
                                    term = F.reshape(F.max(F.absolute(attn_d_grad), axis=1), (attn_d_grad.shape[0], 1, -1))
                                    term = F.broadcast_to(term, attn_d_grad.shape)
                                    attn_d_grad = attn_d_grad / (term + 1e-12)
                                elif args.use_softmax == -3:
                                    term = F.reshape(F.min(F.absolute(attn_d_grad), axis=1), (attn_d_grad.shape[0], 1, -1))
                                    term = F.broadcast_to(term, attn_d_grad.shape)
                                    attn_d_grad = attn_d_grad / (term + 1e-12)
                                elif args.use_softmax == -11:
                                    attn_d_grad = F.softmax(F.normalize(attn_d_grad, axis=1) * args.soft_int_final, axis=1)

                                if args.max_attn:
                                    attn_d_grad_data = attn_d_grad.data if isinstance(attn_d_grad, chainer.Variable) else attn_d_grad
                                    if args.max_attn_type == 0:
                                        max_flags = xp.max(attn_d_grad_original, axis=1, keepdims=True) == attn_d_grad_original
                                        attn_d_grad = xp.where(max_flags, 1., 0.).astype('f')
                                    if args.max_attn_type == -1:
                                        max_flags = xp.max(attn_d_grad_original, axis=1, keepdims=True) == attn_d_grad_original
                                        plus_flags = attn_d_grad_original > 0.0
                                        attn_d_grad = xp.where(max_flags * plus_flags, 1., 0.).astype('f')
                                    elif args.max_attn_type == 1:
                                        max_flags = xp.max(attn_d_grad_original, axis=1, keepdims=True) == attn_d_grad_original
                                        attn_d_grad = xp.where(max_flags, F.normalize(attn_d_grad, axis=1).data, 0.0)
                                    elif args.max_attn_type == 2:
                                        max_flags = xp.max(attn_d_grad_original, axis=1, keepdims=True) == attn_d_grad_original
                                        attn_d_grad = xp.where(max_flags, attn_d_grad_data, 0.0)
                                    elif args.max_attn_type == -2:
                                        max_flags = xp.max(attn_d_grad_original, axis=1, keepdims=True) == attn_d_grad_original
                                        plus_flags = attn_d_grad_original > 0.0
                                        attn_d_grad = xp.where(max_flags * plus_flags, attn_d_grad_data, 0.0)
                                    elif args.max_attn_type == 3:
                                        max_flags = xp.max(attn_d_grad_original, axis=1, keepdims=True) == attn_d_grad_original
                                        attn_d_grad = xp.where(max_flags, attn_d_grad_data, 0.0)
                                    elif args.max_attn_type == 4:
                                        max_flags = xp.max(attn_d_grad_original, axis=1, keepdims=True) == attn_d_grad_original
                                        attn_d_grad = xp.where(max_flags, 1., 0.).astype('f')
                                        attn_d_grad = attn_d_grad * attn_d_grad_norm[..., None]
                                    elif args.max_attn_type == -10:
                                        max_flags_idx = xp.argmax(xp.random.gumbel(size=attn_d_grad_data.shape) + attn_d_grad_data, axis=1).reshape(-1)
                                        max_flags = xp.eye(attn_d_grad_data.shape[1])[max_flags_idx].reshape((attn_d_grad_data.shape[0], attn_d_grad_data.shape[1], 1))
                                        plus_flags = attn_d_grad_data > 0.0
                                        attn_d_grad = xp.where(max_flags * plus_flags, 1., 0.).astype('f')


                                attn_d = F.broadcast_to(attn_d_grad, diff_data_here.shape).data
                                d_vat = xp.sum(attn_d * diff_data_here, axis=1)
                                if args.use_plus_d:
                                    d_vat += xp.sum(d_plus_grad, axis=1)
                            else:
                                d_vat = model.d_var.grad

                                if args.fixed_d:
                                    d_vat = model.xs_noise.grad

                        elif args.use_semi_pred_adv:
                            pred_y = xp.argmax(output_vat.data, axis=1).astype(xp.int32)
                            loss_pred_adv = F.softmax_cross_entropy(
                                output_vat, pred_y, normalize=True)
                            if args.use_nn_term:
                                nn_terms = model.nn_terms
                                loss_pred_adv += nn_terms
                            model.cleargrads()
                            loss_pred_adv.backward()
                            d_vat = model.d_var.grad


                    output_vat = model(_x, _length, d=d_vat)
                    if args.use_semi_vat:
                        loss_vat = net.kl_loss(xp, output_original.data, output_vat)
                    elif args.use_semi_pred_adv:
                        loss_vat = F.softmax_cross_entropy(output_vat, pred_y, normalize=True)
                    # print 'loss:', loss
                    # print 'loss_vat:', loss_vat
                    loss += loss_vat
            else:

                output = model(x, x_length)
                if args.use_crf:
                    loss_crf, predicted_list, y_list_sorted = model.crf_loss(output, x_length, y, ign_loss=False)
                    loss += loss_crf

                else:
                    if args.sent_loss_usual:
                        loss_sent = F.softmax_cross_entropy(output, y, reduce='no')
                        loss_sent_list = F.split_axis(loss_sent, np.cumsum(x_length)[:-1], axis=0)
                        loss_sent_list = F.concat(loss_sent_list, axis=0)
                        loss += F.sum(loss_sent_list) / loss_sent_list.shape[0]
                    else:
                        loss += F.softmax_cross_entropy(output, y, normalize=True)

            if args.use_crf:
                predict_flat = xp.concatenate(predicted_list, axis=0)
                y_flat = xp.concatenate(y_list_sorted, axis=0)
                correct_cnt += xp.sum(predict_flat == y_flat)


                if args.use_seq_labeling:
                    y_np.append(to_cpu(y_flat))
                    predicted_np.append(to_cpu(predict_flat))
            else:
                predict = xp.argmax(output.data, axis=1)
                correct_cnt += xp.sum(predict == y)

                if args.use_seq_labeling:
                    y_np.append(to_cpu(y))
                    predicted_np.append(to_cpu(predict))
            total_cnt += len(y)

            # update
            if args.adv_mode == 0:
                model.cleargrads()
                loss.backward()
                # print 'loss:', loss.data
                opt.update()

            if args.alpha_decay > 0.0:
                if args.use_exp_decay:
                    opt.hyperparam.alpha = (base_alpha) * (args.alpha_decay**global_step)
                else:
                    opt.hyperparam.alpha *= args.alpha_decay  # 0.9999

            sum_loss += loss.data

        if args.accuracy_f1:
            y_np = np.concatenate(y_np, axis=0)
            predicted_np = np.concatenate(predicted_np, axis=0)
            # precision, recall, f1_score, support = precision_recall_fscore_support(y_np, predicted_np)
            # f1_score_val = f1_score_func(y_np, predicted_np, average='binary', pos_label=0)

            precision, recall, f1_score, support = precision_recall_fscore_support(y_np, predicted_np, average='binary', beta=0.5, pos_label=0)
            log_str = '       precision:{}, recall:{}, f1_score:{}'.format(str(precision), str(recall), str(f1_score))
            logging.info(log_str)
            # precision, recall, f1_score, support = precision_recall_fscore_support(y_np, predicted_np, average='micro')
            f1_score_val = np.average(f1_score)

            accuracy = f1_score_val
        else:
            accuracy = correct_cnt / total_cnt
        accuracy = accuracy * 100.0

        log_str = 'sum_loss: {}, cost_e: {}, cost_z: {}, cost_sparsity: {}'.format(
            sum_loss / N, sum_loss_label / N, sum_loss_z / N, sum_loss_z_sparse / N)
        logging.info(' [train] {}'.format(log_str))
        eps_statics, sim_statics, sim_most_statics, auto_scale_statics, d_original_norm, d_norm, impscore, norm_diff, norm_diff_min, norm_noise = model.get_statics()
        logging.info(' [train] eps:{}, sim:{}, most_sim:{}, auto_scale:{}'.format(
            eps_statics, sim_statics, sim_most_statics, auto_scale_statics))
        logging.info(' [train] n(d_ori):{}, n(d):{}, impscore:{}, n(diff):{}, n(diff_min):{}, norm_noise:{}'.format(
            d_original_norm, d_norm, impscore, norm_diff, norm_diff_min, norm_noise))


        logging.info(' [train] apha:{}, global_step:{}'.format(opt.hyperparam.alpha, global_step))
        logging.info(' [train] accuracy:{}'.format(accuracy))

        if args.use_rational == 1:
            # def cal_rate(mat):
                # return xp.sum(mat.data) / float(mat.shape[0])
            # selected_mask_list = model.selected_mask_list
            # avg_rate = [cal_rate(_) for _ in selected_mask_list]
            avg_rate = avg_rate / avg_rate_num
            logging.info(' [train] length:' + str(avg_rate))

        model.set_train(False)
        # dev
        dev_accuracy, dev_avg_rate = evaluate(dev_x, dev_x_len, dev_y)

        log_str = ' [dev] accuracy:{}, length:{}'.format(str(dev_accuracy), str(dev_avg_rate))
        logging.info(log_str)

        if args.debug or args.debug_small:
            continue

        # test
        test_accuracy, test_avg_rate = evaluate(test_x, test_x_len, test_y)
        log_str = ' [test] accuracy:{}, length:{}'.format(str(test_accuracy), str(test_avg_rate))
        logging.info(log_str)

        # save_flag = dev_accuracy > 80.0 and dev_avg_rate < 0.50
        last_epoch_flag = args.n_epoch - 1 == epoch
        save_flag = args.save_flag or (args.save_last and last_epoch_flag)
        if prev_dev_accuracy < dev_accuracy and save_flag:

            logging.info(' => '.join([str(prev_dev_accuracy), str(dev_accuracy)]))
            result_str = 'dev_acc_' + str(dev_accuracy)
            result_str += '_test_acc_' + str(test_accuracy)
            model_filename = './models/' + '_'.join([args.save_name,
                                                     str(epoch), result_str])
            # if len(sentences_train_list) == 1:
            serializers.save_hdf5(model_filename + '.model', model)

            prev_dev_accuracy = dev_accuracy

        nn_flag = args.update_nearest_epoch > 0 and (epoch % args.update_nearest_epoch == 0)
        if all_nn_flag and nn_flag and args.online_nn == 0:
            model.cleargrads()
            x = None
            x_length = None
            y = None
            model.all_nearest_words(top_k=args.nn_k)


if __name__ == '__main__':
    main()
