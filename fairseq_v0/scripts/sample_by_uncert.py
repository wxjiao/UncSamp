# By wxjiao, 2020-May
# Functions: Sample monolingual data according to the uncertainty distribution.
# Step: 3
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


# Read token probability
def read_rank(path):
    sorted_id = []
    sorted_p = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            i, p = line.strip('\n').split()
            i, p = int(i), float(p)
            sorted_id.append(i)
            sorted_p.append(p)
    sorted_id_np = np.array(sorted_id)
    sorted_p_np = np.array(sorted_p)
    return sorted_id_np, sorted_p_np


def sample(sorted_uc, size, uc_ref, uc_pen=0, beta=1, replace=False, seed=1):
    if uc_pen == 0:
        new_uc = sorted_uc
    else:
        new_uc = [uc_i if uc_i <= uc_ref else max(2*uc_ref-uc_i, 0) for uc_i in sorted_uc]
        new_uc = np.array(new_uc)
    norm_uc = new_uc**beta / np.sum(new_uc**beta)
    idx_uc = np.array(range(len(new_uc)))
    np.random.seed(seed)
    select_idx = np.random.choice(idx_uc, size, p=norm_uc, replace=replace)
    return select_idx


def write_train(data, selected, path):
    with open(path, 'w', encoding='utf-8') as file:
        for i in tqdm(selected):
            file.write(data[i])


if __name__ == '__main__':
    DISK_SAVE = "[Your project location]"
    path_sorted_id = DISK_SAVE + "/monolingual/mono_en_2011_2019_ende/bpe.news_196M.ende.en.ids_sorted"
    sorted_id, sorted_uc = read_rank(path_sorted_id)
    size = 40000000
    select_idx = sample(sorted_uc, size, uc_ref=2.90, uc_pen=1, beta=2, replace=False, seed=1)
    path_src = DISK_SAVE + "/monolingual/mono_en_2011_2019_ende/bpe.news_196M.ende.en.sorted"
    src_bt = read_train(path_src)
    # Save according to the order
    write_train(src_bt, select_idx, DISK_SAVE + "/monolingual/mono_en_2011_2019_ende/bpe.news_196M.ende.en.selected")

