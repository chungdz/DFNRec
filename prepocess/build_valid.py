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
        imp_pos_list = []
        imp_neg_list = []
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
    f_dev_beh = os.path.join("data", args.fsamples)
    df = pd.read_csv(f_dev_beh, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
    news_info = pickle.load(open('data/news_n.pkl', 'rb'))
    user_info = pickle.load(open('data/user_n.pkl', 'rb'))

    subdf_len = math.ceil(len(df) / args.processes)
    cut_indices = [x * subdf_len for x in range(1, args.processes)]
    dfs = np.split(df, cut_indices)

    processes = []
    for i in range(args.processes):
        output_path = os.path.join("data", args.fout,  "dev-{}.npy".format(i))
        p = mp.Process(target=build_examples, args=(
            i, args, dfs[i], news_info, user_info, output_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path options.
    parser.add_argument("--fsamples", default="dev/behaviors.tsv", type=str,
                        help="Path of the dev samples file.")
    parser.add_argument("--fout", default="raw", type=str,
                        help="Path of the output dir.")
    parser.add_argument("--neg_count", default=4, type=int)
    parser.add_argument("--processes", default=40, type=int, help="Processes number")
    parser.add_argument("--pos_hist_length", default=30, type=int)
    parser.add_argument("--neg_hist_length", default=30, type=int)
    parser.add_argument("--unclicked_hist_length", default=30, type=int)

    args = parser.parse_args()

    main(args)

