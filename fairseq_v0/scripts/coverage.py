# By wxjiao, 2020-May
# Functions: Coverage of sentence pairs for both source and target sides.
#
import os
import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 300


# Read in bpe data
def read_train(path):
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines


def comp_coverage(src_data, tgt_data, ali_data):
    src_align, tgt_align = [], []
    for idx in tqdm(range(len(src_data))):
        src = src_data[idx].strip('\n').split()
        tgt = tgt_data[idx].strip('\n').split()
        ali = ali_data[idx].strip('\n').split()
        src_ali, tgt_ali = [], []
        for al in ali:
            als, alt = al.split('-')
            src_ali.append(als)
            tgt_ali.append(alt)
        src_align.append(len(set(src_ali)) / len(src))
        tgt_align.append(len(set(tgt_ali)) / len(tgt))
    cov_src, cov_tgt = np.mean(src_align), np.mean(tgt_align)
    return cov_src, cov_tgt


if __name__ == "__main__":
    DISK_SAVE = "[Your project location]"
    path_src = DISK_SAVE + "/uncertainty/wmt19_en_de_bitext/train.en"
    path_tgt = DISK_SAVE + "/uncertainty/wmt19_en_de_bitext/train.de"
    path_ali = DISK_SAVE + "/uncertainty/wmt19_en_de_bitext/train.sym_align"
    path_res = DISK_SAVE + "/uncertainty/wmt19_en_de_bitext/coverage.log"
    # Read data
    print("Start to read data.")
    src_data = read_train(path_src)
    tgt_data = read_train(path_tgt)
    ali_data = read_train(path_ali)
    info_data = "Lengths: src_data {}, tgt_data {}, ali_data {}\n".format(len(src_data), len(tgt_data), len(ali_data))
    # Calculate
    cov_src, cov_tgt = comp_coverage(src_data, tgt_data, ali_data)
    info_cov = "Coverage: src {}, tgt {}\n".format(cov_src, cov_tgt)

    with open(path_res, 'w', encoding='utf-8') as file:
        file.write(info_data)
        file.write(info_cov)
