import numpy as np
import argparse
import json
import math
import os

parser = argparse.ArgumentParser()

parser.add_argument("--fsamples", default="adressa/raw/dev", type=str,
                    help="Path of the training samples file.")
parser.add_argument("--split", default=10, type=int,
                    help="Processes number")
parser.add_argument('--filenum', type=int, default=10)

cfg = parser.parse_args()

data_list = []

for i in range(cfg.filenum):
    file_name = "{}-{}.npy".format(cfg.fsamples, i)
    data_list.append(np.load(file_name))
    os.remove(file_name)

sub_len = data_list[0].shape[0]
datanp = np.concatenate(data_list, axis=0)

np.save("{}-{}.npy".format(cfg.fsamples, 0), datanp[:sub_len])
np.save("{}-{}.npy".format(cfg.fsamples, 1), datanp[sub_len:])
