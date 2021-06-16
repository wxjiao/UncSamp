# By wxjiao, 2020-May
# Functions: Word rarity of sentences. Length of sentences is also provided.
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
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            w, f = line.strip('\n').split()
            word_dict[w] = int(f)
    rarity_dict = dict()
    tot_f = np.sum(list(word_dict.values()))
    for k,v in word_dict.items():
        wr = - np.log(float(v) / tot_f)
        rarity_dict[k] = wr
    return word_dict, rarity_dict


def save_dict(count_dict, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for k,v in count_dict.items():
            file.write("{}\t{}\n".format(k, v))


# Read in bpe data
def calc_rarity(path, rarity_dict):
    wr_sents = []
    len_sents = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            ws = line.strip('\n').split()
            wrs = [rarity_dict[w] if w in rarity_dict.keys() else max(rarity_dict.values()) for w in ws]
            wr_sent = np.mean(wrs)
            wr_sents.append(wr_sent)
            len_sents.append(len(wrs))
    wr_sents_mean = np.mean(wr_sents)
    len_sents_mean = np.mean(len_sents)
    return wr_sents, wr_sents_mean, len_sents_mean


if __name__ == "__main__":
    DISK_SAVE = "[Your project location]"
    path_wr_dict = DISK_SAVE + "/wordrarity/wmt19_en_de_bitext/dict.en.txt"
    _, rarity_dict = read_dict(path_wr_dict)
    save_dict(rarity_dict, path_wr_dict + ".rarity")
    info_dict = "Rarity dictionay: length {}\n".format(len(rarity_dict))
    print(info_dict)
    path_data = DISK_SAVE + "/wordrarity/wmt19_en_de_bitext/train.en"
    path_res = DISK_SAVE + "/wordrarity/wmt19_en_de_bitext/rarity.log"
    # Read data
    _, wr_sents_mean, len_sents_mean = calc_rarity(path_data, rarity_dict)
    info_data = "Mono corpus: rarity {}, length {}\n".format(wr_sents_mean, len_sents_mean)

    with open(path_res, 'w', encoding='utf-8') as file:
        file.write(info_dict)
        file.write(info_data)
