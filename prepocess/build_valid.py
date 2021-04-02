import os
import json
import pickle
import argparse
import math
import pandas as pd
import numpy as np
import multiprocessing as mp
import random
from tqdm import tqdm

random.seed(7)

def build_examples(rank, args, df, news_info, user_info, fout):
    data_list = []
    input_len = 2 + 2 + (1 + args.pos_hist_length + args.neg_hist_length + args.unclicked_hist_length)

    for imp_id, uid, imp in tqdm(df[["id", "uid", "imp"]].values, total=df.shape[0]):
        if uid not in user_info:
            continue
        his_idx_list = user_info[uid]["pos"] + user_info[uid]["neg"] + user_info[uid]["unclicked"]

        imp_list = str(imp).split(' ')

        has_one = False
        has_zero = False
        for impre in imp_list:
            arr = impre.split('-')
            curn = news_info[arr[0]]['idx']
            label = int(arr[1])
            if label == 1:
                has_one = True
            else:
                has_zero = True
        
        if not (has_one and has_zero):
            continue
        
        for impre in imp_list:
            arr = impre.split('-')
            curn = news_info[arr[0]]['idx']
            label = int(arr[1])
            
            new_row = []
            new_row.append(int(imp_id))
            new_row.append(label)
            # idx
            new_row.append(user_info[uid]['idx'])
            new_row.append(curn)
            # title
            new_row.append(curn)
            new_row += his_idx_list
            assert(len(new_row) == input_len)
            data_list.append(new_row)
    
    datanp = np.array(data_list, dtype=int)
    np.save(fout, datanp)
    print(datanp.shape)

def main(args):
    f_dev_beh = os.path.join(args.root, args.fsamples)
    if args.dataset == 'MIND':
        df = pd.read_csv(f_dev_beh, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
    else:
        df = pd.read_csv(f_dev_beh, sep="\t", encoding="utf-8", names=["id", "uid", "imp"])
    news_info = pickle.load(open('{}/news_n.pkl'.format(args.root), 'rb'))
    user_info = pickle.load(open('{}/user_n.pkl'.format(args.root), 'rb'))

    subdf_len = math.ceil(len(df) / args.processes)
    cut_indices = [x * subdf_len for x in range(1, args.processes)]
    dfs = np.split(df, cut_indices)

    processes = []
    for i in range(args.processes):
        output_path = os.path.join(args.root, args.fout,  "{}-{}.npy".format(args.type, i))
        p = mp.Process(target=build_examples, args=(
            i, args, dfs[i], news_info, user_info, output_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path options.
    parser.add_argument("--fsamples", default="dev/target_behaviors.tsv", type=str,
                        help="Path of the dev samples file.")
    parser.add_argument("--fout", default="raw", type=str,
                        help="Path of the output dir.")
    parser.add_argument("--neg_count", default=4, type=int)
    parser.add_argument("--processes", default=40, type=int, help="Processes number")
    parser.add_argument("--pos_hist_length", default=30, type=int)
    parser.add_argument("--neg_hist_length", default=30, type=int)
    parser.add_argument("--unclicked_hist_length", default=30, type=int)

    parser.add_argument("--root", default="data", type=str)
    parser.add_argument("--dataset", default="MIND", type=str)
    parser.add_argument("--type", default='dev', type=str)
    args = parser.parse_args()

    main(args)

