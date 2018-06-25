#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sklearn
from string import ascii_letters
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

import pickle
import sys
import os

def vis(items_obj, figsave='a'):
    vis_lists, x, y = items_obj

    original_words = []
    nn_words_lists = []
    attn_alphas = []
    print('vis_lists:', len(vis_lists))
    print('x:', x)
    print('y:', y)
    max_diff_norm = 0.0
    diff_norm_sum = 0.0
    for item in vis_lists:
        [r_i, original_word, rep_word, max_scalar, attn_d_bf, nn_words, gra_sc, flag, sims, diff_norm_scala, d_data_scala] = item
        print('[pickle] original_word:', original_word)
        original_words.append(original_word)

        if len(gra_sc.shape) == 0:
            gra_sc = gra_sc[..., None]

        print('gra_sc:', gra_sc.shape, gra_sc)
        # diff_norm_scala is the |diff|
        print('diff_norm_scala:', diff_norm_scala.shape, diff_norm_scala)
        max_diff_norm = max(max_diff_norm, diff_norm_scala)
        diff_norm_sum += diff_norm_scala
        score = (gra_sc / diff_norm_scala)
        print('[before] score:', score)

        score = score * args.xi_var
        print('[after] score:', score)

        attn_alphas.append(score)
        max_idx = np.argmax(sims)
        max_word = nn_words.split(',')[max_idx]
        nn_words_lists.append(max_word)

    diff_norm_avg = diff_norm_sum / len(vis_lists)
    data = np.array(attn_alphas)
    # data = data / max(data)
    # data = data * 0.46
    print('data:', data.shape)
    # width = 30
    width = 4
    plt.figure(figsize=(width, 15))
    annot = np.array([nn_words_lists[_i].split(',') for _i in range(data.shape[0])])
    annot = np.reshape(annot, data.shape)
    print('annot:', annot.shape)
    print(annot)
    vmin = 0.0

    if int(y[0]) == 0:
        # Negative to Positive
        cmap = 'Reds'
    else:
        # Positive to Negative
        cmap = 'Blues'

    print('original_words:', original_words)

    sns.heatmap(data, yticklabels=original_words, annot=annot, fmt='s', cmap=cmap, cbar=True, vmin=vmin, vmax=diff_norm_avg, xticklabels=0, annot_kws={"size": 14})
    plt.yticks(rotation=0)
    plt.tight_layout()

    words_str = '_'.join(original_words)
    save_path = figsave + words_str[:10] + '_' + str(y) + str(flag) + '.png'
    print('fig_path:', save_path)
    plt.savefig(save_path)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--pickle_filename', dest='pickle_filename',
        type=str, default='', help='pickle_filename')
    parser.add_argument('--savefig_dir', dest='savefig_dir',
        type=str, default='figs', help='savefig_dir')
    parser.add_argument('--xi_var', dest='xi_var',
        type=float, default=15.0, help='xi_var')
    args = parser.parse_args()

    filename = args.pickle_filename
    savefig_name = args.savefig_dir
    os.makedirs(savefig_name, exist_ok=True)

    # load pickle
    with open(filename, mode='rb') as f:
        items = pickle.load(f)

    print('items:', len(items))

    for s, items_obj in enumerate(items):
        figsave = savefig_name + '/' + str(s)
        vis(items_obj, figsave)
