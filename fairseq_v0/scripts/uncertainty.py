# By wxjiao, 2020-May
# Fucntions: Uncertainty at word-level (uncertainty dict) and sentence-level. Both source-side and target-side uncertainty are provided.
# Step: 1
#
import os
import time
import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 300


def read_dict(path):
    word_dict = dict()
    idx = 0
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            w, f = line.strip('\n').split()
            word_dict[w] = idx
            idx += 1
    return word_dict


def save_dict(count_dict, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for k,v in count_dict.items():
            file.write("{}\t{}\n".format(k, v))

# Read in bpe data
def read_train(path):
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines


# For each sentence pair, count alignments of target tokens conditioned on each source token
def count_align(src_list, tgt_list, align_list, align_dict, align_dict_rev):
    for al in align_list:
        als, alt = al.split('-')
        als, alt = int(als), int(alt)
        s_tok, t_tok = src_list[als], tgt_list[alt]
        # Forward align_dict
        if s_tok not in align_dict.keys():
            align_dict[s_tok] = dict()
        if t_tok not in align_dict[s_tok].keys():
            align_dict[s_tok][t_tok] = 1
        else:
            align_dict[s_tok][t_tok] += 1
        # Reversed align_dict
        if t_tok not in align_dict_rev.keys():
            align_dict_rev[t_tok] = dict()
        if s_tok not in align_dict_rev[t_tok].keys():
            align_dict_rev[t_tok][s_tok] = 1
        else:
            align_dict_rev[t_tok][s_tok] += 1
    return align_dict, align_dict_rev


# Obatin the align_dict of the whole corpus
def corpus_align(src_data, tgt_data, ali_data):
    align_dict, align_dict_rev = dict(), dict()
    for idx in range(len(src_data)):
        src = src_data[idx].strip('\n').split()
        tgt = tgt_data[idx].strip('\n').split()
        ali = ali_data[idx].strip('\n').split()
        align_dict, align_dict_rev = count_align(src, tgt, ali, align_dict, align_dict_rev)
    return align_dict, align_dict_rev
    

# Compute uncertainty by the align_dict
def comp_uncertainty(align_dict):
    H = []
    W = []
    uncert_dict = dict()
    for k,v in align_dict.items():
        tot_k = sum(v.values())    # total counts of alignment to word k
        h = 0
        for kk,vv in v.items():
            p = vv / float(tot_k)  # p for k-kk alignment
            h = h - p * np.log(p)
        H.append(h)
        W.append(tot_k)
        uncert_dict[k] = h
    np_H, np_W = np.array(H), np.array(W)
    wgt_H = np.mean(np_H * np_W) / np.mean(np_W)
    avg_H = np.mean(np_H)
    return avg_H, wgt_H, uncert_dict


if __name__ == "__main__":
    DISK_SAVE = "[Your project location]"
    # Dictionary path
    path_src_dict = DISK_SAVE + "/uncertainty/wmt19_en_de_bitext/dict.en.txt"
    path_tgt_dict = DISK_SAVE + "/uncertainty/wmt19_en_de_bitext/dict.de.txt"
    src_dict, tgt_dict = read_dict(path_src_dict), read_dict(path_tgt_dict)
    # Data path
    path_src = DISK_SAVE + "/uncertainty/wmt19_en_de_bitext/train.en"
    path_tgt = DISK_SAVE + "/uncertainty/wmt19_en_de_bitext/train.de"
    path_ali = DISK_SAVE + "/uncertainty/wmt19_en_de_bitext/train.sym_align"
    path_res = DISK_SAVE + "/uncertainty/wmt19_en_de_bitext/uncertainty.log"
    # Read data
    print("Start to read data.")
    src_data = read_train(path_src)
    tgt_data = read_train(path_tgt)
    ali_data = read_train(path_ali)
    info_data = "Lengths: src_data {}, tgt_data {}, ali_data {}\n".format(len(src_data), len(tgt_data), len(ali_data))
    print(info_data)
    # Calculate
    align_dict, align_dict_rev = corpus_align(src_data, tgt_data, ali_data) # Word alignments from source to target and reversed.
    avg_H, wgt_H, uncert_dict_fwd = comp_uncertainty(align_dict)
    save_dict(uncert_dict_fwd, path_src + ".uncert")
    info_uncert_fwd = "Forward Uncertainty: mean {}, weighted {}\n".format(avg_H, wgt_H)
    print(info_uncert_fwd)
    avg_H_rev, wgt_H_rev, uncert_dict_bwd = comp_uncertainty(align_dict_rev)
    save_dict(uncert_dict_bwd, path_tgt + ".uncert")
    info_uncert_bwd = "Reversed Uncertainty: mean {}, weighted {}\n".format(avg_H_rev, wgt_H_rev)
    print(info_uncert_bwd)

    with open(path_res, 'w', encoding='utf-8') as file:
        file.write(info_dict)
        file.write(info_data)
        file.write(info_align)
        file.write(info_uncert_fwd)
        file.write(info_uncert_bwd)
