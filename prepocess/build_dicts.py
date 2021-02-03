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

parser.add_argument("--title_len", default=30, type=int,
                    help="Max length of the title.")

args = parser.parse_args()

data_path = 'data'
max_title_len = args.title_len

print("Loading news info")
f_train_news = os.path.join(data_path, "train/news.tsv")
f_dev_news = os.path.join(data_path, "dev/news.tsv")

print("Loading training news")
all_news = pd.read_csv(f_train_news, sep="\t", encoding="utf-8",
                        names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
                        quoting=3)

print("Loading dev news")
dev_news = pd.read_csv(f_dev_news, sep="\t", encoding="utf-8",
                        names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
                        quoting=3)
all_news = pd.concat([all_news, dev_news], ignore_index=True)


news_dict = {}
word_dict = {'<pad>': 0}
word_idx = 1
news_idx = 1
for n, title, topic in all_news[['newsid', "title", "subcate"]].values:
    news_dict[n] = {}
    news_dict[n]['idx'] = news_idx
    news_dict[n]['clicked'] = set()
    news_dict[n]['neighbor'] = []
    news_idx += 1

    tarr = removePunctuation(title).split()
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
    
    news_dict[n]['title'] = wid_arr[:max_title_len]

## paddning news for history
news_dict['<his>']= {}
news_dict['<his>']['idx'] = 0
news_dict['<his>']['title'] = list(np.zeros(max_title_len))
news_dict['<his>']['clicked'] = set()
news_dict['<his>']['neighbor'] = []

print('all word', len(word_dict))
print('all news', len(news_dict))

print("Loading behaviors info")
f_his_beh = os.path.join(data_path, "train/his_behaviors.tsv")
f_target_beh = os.path.join(data_path, "train/target_behaviors.tsv")
f_dev_beh = os.path.join(data_path, "dev/behaviors.tsv")

print("Loading his beh")
his_beh = pd.read_csv(f_his_beh, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
print("Loading target beh")
target_beh = pd.read_csv(f_target_beh, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
print("Loading dev beh")
dev_beh = pd.read_csv(f_dev_beh, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])

target_ids = set(pd.unique(target_beh['uid']))
dev_ids = set(pd.unique(dev_beh['uid']))
live_ids = target_ids | dev_ids
print('live ids', len(live_ids))

user_dict = {}
user_idx = 0
for uid, imp in tqdm(his_beh[['uid', 'imp']].values, total=his_beh.shape[0], desc='history behavior'):
    if uid not in live_ids:
        continue

    if uid not in user_dict:
        user_dict[uid] = {"pos": [], "neg": [], "idx": user_idx, 'clicked': []}
        user_idx += 1
    
    imp_list = str(imp).split(' ')
    for impre in imp_list:
        arr = impre.split('-')
        curn = news_dict[arr[0]]['idx']
        label = int(arr[1])
        if label == 0:
            user_dict[uid]["neg"].append(curn)
        elif label == 1:
            user_dict[uid]["pos"].append(curn)
            user_dict[uid]['clicked'].append(arr[0])
        else:
            raise Exception('label error!')

print('user num', len(user_dict))
# build graph dict
for uid, uinfo in tqdm(user_dict.items(), desc='build graph', total=len(user_dict)):
    
    his_list = uinfo["clicked"]
    for h in his_list:
        news_dict[h]['clicked'].add(uid)

build_word_embeddings(word_dict, 'data/glove.840B.300d.txt', 'data/emb.npy')
build_news_embeddings(news_dict, 'data/news_info.npy')
pickle.dump(user_dict, open('data/user.pkl', 'wb'))
pickle.dump(news_dict, open('data/news.pkl', 'wb'))
json.dump(word_dict, open('data/word.json', 'w', encoding='utf-8'))

