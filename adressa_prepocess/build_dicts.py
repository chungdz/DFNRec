import os
import json
import pickle
import argparse
import re
import pandas as pd
import numpy as np
from prepocess.embd import build_word_embeddings, build_news_embeddings
from tqdm import tqdm

def parse_ent_list(x):
    if x.strip() == "":
        return ''

    return ' '.join([k["WikidataId"] for k in json.loads(x)])

punctuation = '!,;:?"\''
def removePunctuation(text):
    text = re.sub(r'[{}]+'.format(punctuation),'',text)
    return text.strip().lower()

parser = argparse.ArgumentParser()

parser.add_argument("--title_len", default=15, type=int,
                    help="Max length of the title.")

args = parser.parse_args()

data_path = 'adressa'
max_title_len = args.title_len

print("Loading news info")
news_dict_raw = json.load(open('adressa/news_dict.json'))

news_dict = {}
word_dict = {'<pad>': 0}
word_idx = 1
news_idx = 1
for nid, ninfo in news_dict_raw.items():
    
    news_dict[nid] = {}
    news_dict[nid]['idx'] = news_idx
    news_dict[nid]['clicked'] = set()
    news_dict[nid]['neighbor'] = []
    news_idx += 1

    tarr = removePunctuation(ninfo).split()
    wid_arr = []
    for t in tarr:
        if t not in word_dict:
            word_dict[t] = word_idx
            word_idx += 1
        wid_arr.append(word_dict[t])
    cur_len = len(wid_arr)
    if cur_len < max_title_len:
        for l in range(max_title_len - cur_len):
            wid_arr.append(0)
    
    news_dict[nid]['title'] = wid_arr[:max_title_len]

## paddning news for history
news_dict['<his>']= {}
news_dict['<his>']['idx'] = 0
news_dict['<his>']['title'] = list(np.zeros(max_title_len))
news_dict['<his>']['clicked'] = set()
news_dict['<his>']['neighbor'] = []

print('all word', len(word_dict))
print('all news', len(news_dict))
assert(len(news_dict) == news_idx)

print("Loading behaviors info")
behaviors_raw = json.load(open('adressa/his_behaviors.json'))

user_dict = {}
for uid, uinfo in tqdm(behaviors_raw.items(), total=len(behaviors_raw), desc='history behavior'):
    
    user_dict[uid] = {"pos": [], "neg": [], 'clicked': [], 'dislike': []}
    
    for nid in uinfo['pos']:
        user_dict[uid]["pos"].append(news_dict[nid]['idx'])
        user_dict[uid]["clicked"].append(nid)
    for nid in uinfo['neg']:
        user_dict[uid]['neg'].append(news_dict[nid]['idx'])
        user_dict[uid]["dislike"].append(nid)

print('user num', len(user_dict))
# build graph dict
for uid, uinfo in tqdm(user_dict.items(), desc='build graph', total=len(user_dict)):
    
    his_list = uinfo["clicked"]
    for h in his_list:
        news_dict[h]['clicked'].add(uid)

build_word_embeddings(word_dict, 'data/glove.840B.300d.txt', 'adressa/emb.npy')
build_news_embeddings(news_dict, 'adressa/news_info.npy')
pickle.dump(user_dict, open('adressa/user.pkl', 'wb'))
pickle.dump(news_dict, open('adressa/news.pkl', 'wb'))
json.dump(word_dict, open('adressa/word.json', 'w', encoding='utf-8'))


