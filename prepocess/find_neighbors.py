import os
import json
import pickle
import argparse
import re
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

D = 5
with open('data/user.pkl', 'rb') as f:
    user_dict = pickle.load(f)
with open('data/news.pkl', 'rb') as f2:
    news_dict = pickle.load(f2)

for n, info in tqdm(news_dict.items(), total=len(news_dict), desc='news neighbor'):
    if len(info['clicked']) < 1:
        continue

    neighbor_news_dict = {}
    for u in info['clicked']:
        cur_nlist = user_dict[u]['clicked']
        for neighbor_n in cur_nlist:
            if neighbor_n not in neighbor_news_dict:
                neighbor_news_dict[neighbor_n] = 1
            else:
                neighbor_news_dict[neighbor_n] += 1
    
    info['neighbor'] = sorted(neighbor_news_dict, key=lambda x: -neighbor_news_dict[x])[:D]


with open('data/news_n.pkl', 'wb') as f4:
    news_dict = pickle.dump(news_dict, f4)
