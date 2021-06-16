# By wxjiao, 2020-May
# Functions: Sort the monolingual data by their uncertainty.
# Step: 2
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


def read_dict(path):
    uncert_dict = dict()
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            w, uc = line.strip('\n').split()
            uncert_dict[w] = float(uc)
    return uncert_dict


def calc_uncertainty(path, uncert_dict):
    uc_sents = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            ws = line.strip('\n').split()
            ucs = [uncert_dict[w] if w in uncert_dict.keys() else 1e-6 for w in ws]
            uc_sent = np.mean(ucs)
            uc_sents.append(uc_sent)
    uc_sents_mean = np.mean(uc_sents)
    return uc_sents, uc_sents_mean


# Calculate sentence-level prob
def scorer(uc_sents):
    ids = list(range(len(uc_sents)))
    print("Sorting sentence pairs by source uncertainty.\n")
    ids = np.array(ids)
    uc_sents = np.array(uc_sents)
    sorted_idx = np.argsort(uc_sents)
    sorted_ids = ids[sorted_idx]
    sorted_uc = uc_sents[sorted_idx]
    return ids, uc_sents, sorted_idx, sorted_ids, sorted_uc


def write_train(data, selected, path):
    with open(path, 'w', encoding='utf-8') as file:
        for i in tqdm(selected):
            file.write(data[i])


if __name__ == '__main__':
    DISK_SAVE = "[Your project location]"
    path_uc_dict = DISK_SAVE + "/monolingual/mono_en_2011_2019_ende/train.en.uncert"
    uncert_dict = read_dict(path_uc_dict)

    path_src = DISK_SAVE + "/monolingual/mono_en_2011_2019_ende/bpe.news_196M.ende.en"
    uc_sents, _ = calc_uncertainty(path_src, uncert_dict)
    sorted_ids_mono, sorted_p_mono = scorer(uc_sents)[-2:]

    # Save sorted data
    src_data = read_train(path_src)
    path_dir = DISK_SAVE + "/monolingual/mono_en_2011_2019_ende"
    # Save according to the order
    write_train(src_data, sorted_ids_mono, path_dir + "/bpe.news_196M.ende.en.sorted")
    with open(path_dir + "/bpe.news_196M.ende.en.ids_sorted", 'w', encoding='utf-8') as f:
        for i in range(0, len(sorted_ids_mono)):
            f.write("{}\t{}\n".format(sorted_ids_mono[i], sorted_p_mono[i]))
