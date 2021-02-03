import os
import json
import pickle
import argparse
import re
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--D", default=5, type=int, help="Max length of the news neighbor")
parser.add_argument("--pos_hist_length", default=30, type=int)
parser.add_argument("--neg_hist_length", default=30, type=int)
parser.add_argument("--unclicked_hist_length", default=30, type=int)

args = parser.parse_args()
D = args.D

with open('data/user.pkl', 'rb') as f:
    user_dict = pickle.load(f)
with open('data/news.pkl', 'rb') as f2:
    news_dict = pickle.load(f2)

has_neighbor = 0
for n, info in tqdm(news_dict.items(), total=len(news_dict), desc='news neighbor'):
    if len(info['clicked']) < 1:
        continue
    has_neighbor += 1
    neighbor_news_dict = {}
    for u in info['clicked']:
        cur_nlist = user_dict[u]['pos']
        for neighbor_n in cur_nlist:
            if neighbor_n not in neighbor_news_dict:
                neighbor_news_dict[neighbor_n] = 1
            else:
                neighbor_news_dict[neighbor_n] += 1
    
    info['neighbor'] = sorted(neighbor_news_dict, key=lambda x: -neighbor_news_dict[x])[:D]

print("There are {} news has neighbor".format(has_neighbor))

for uid, uinfo in tqdm(user_dict.items(), total=len(user_dict), desc='pad history'):

    # unclicked
    neg_list = uinfo["dislike"]
    unclicked_list = []
    for neg_news in neg_list:
        neg_neighbor_list = news_dict[neg_news]['neighbor']
        unclicked_list = unclicked_list + neg_neighbor_list
    uinfo["unclicked"] = unclicked_list

    pos_len = len(uinfo["pos"])
    if pos_len < args.pos_hist_length:
        for _ in range(args.pos_hist_length - pos_len):
            uinfo["pos"].insert(0, news_dict['<his>']['idx'])
    else:
        uinfo["pos"] = uinfo["pos"][-args.pos_hist_length:]
    
    neg_len = len(uinfo["neg"])
    if neg_len < args.neg_hist_length:
        for _ in range(args.neg_hist_length - neg_len):
            uinfo["neg"].insert(0, news_dict['<his>']['idx'])
    else:
        uinfo["neg"] = uinfo["neg"][-args.neg_hist_length:]
    
    unclicked_len = len(uinfo['unclicked'])
    if unclicked_len < args.unclicked_hist_length:
        for _ in range(args.unclicked_hist_length - unclicked_len):
            uinfo["unclicked"].insert(0, news_dict['<his>']['idx'])
    else:
        uinfo["unclicked"] = random.sample(uinfo["unclicked"], args.unclicked_hist_length)


with open('data/news_n.pkl', 'wb') as f4:
    news_dict = pickle.dump(news_dict, f4)
with open('data/user_n.pkl', 'wb') as f4:
    user_dict = pickle.dump(user_dict, f4)
