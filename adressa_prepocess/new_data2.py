import json
import datetime
import os
import pickle
from tqdm import tqdm, trange
import numpy as np
import random
import pandas as pd

news_dict = json.load(open('adressa/news_raw.json', 'r'))
all_beh_df = pd.read_csv('adressa/raw_behaviors.tsv', sep='\t')
neg_count = 4
random.seed(7)
# add news idx
news_simplified_dict = {}
neighbor_dict = {}
for nid, ninfo in tqdm(news_dict.items(), total=len(news_dict), desc='add news index'):
    news_simplified_dict[ninfo['idx']] = ninfo['title']
    neighbor_dict[ninfo['idx']] = set()

# his
print('parse his')
begin = datetime.date(2017, 1, 1).strftime('%Y%m%d')
end = datetime.date(2017, 2, 28).strftime('%Y%m%d')
his_user_dict = {}

his_df = all_beh_df[all_beh_df['datetime'] <= int(end)]
for uid, nid, at in his_df[['uid', 'nid', 'activeTime']].values:
    if uid not in his_user_dict:
        his_user_dict[uid] = []
    his_user_dict[uid].append((nid, at))

# train
print('parse train')
begin = datetime.date(2017, 3, 1).strftime('%Y%m%d')
end = datetime.date(2017, 3, 21).strftime('%Y%m%d')
train_user_dict = {}

train_df = all_beh_df[(all_beh_df['datetime'] >= int(begin)) & (all_beh_df['datetime'] <= int(end))]
for uid, nid, at in train_df[['uid', 'nid', 'activeTime']].values:
    if uid not in train_user_dict:
        train_user_dict[uid] = []
    train_user_dict[uid].append((nid, at))

# valid
print('parse valid')
begin = datetime.date(2017, 3, 22).strftime('%Y%m%d')
end = datetime.date(2017, 3, 31).strftime('%Y%m%d')
valid_user_dict = {}

valid_df = all_beh_df[all_beh_df['datetime'] >= int(begin)]
for uid, nid, at in valid_df[['uid', 'nid', 'activeTime']].values:
    if uid not in valid_user_dict:
        valid_user_dict[uid] = []
    valid_user_dict[uid].append((nid, at))

# remove clicks lower than 5
uidx = 0
filter_user = {}
for uid, uinfo in his_user_dict.items():
    if len(uinfo) > 4:
        filter_user[uid] = {}
        filter_user[uid]['his'] = uinfo
        filter_user[uid]['train'] = []
        filter_user[uid]['valid'] = []
        filter_user[uid]['test'] = []
        filter_user[uid]['uidx'] = 'U' + str(uidx)
        uidx += 1
# filter train user
train_num = 0
for uid, uinfo in train_user_dict.items():
    if uid in filter_user:
        filter_user[uid]['train'] = uinfo
        train_num += len(uinfo)
# filter valid user
valid_num = 0
for uid, uinfo in valid_user_dict.items():
    if uid in filter_user:
        filter_user[uid]['valid'] = uinfo
        valid_num += len(uinfo)

# active time check
active_time = sorted(all_beh_df['activeTime'])
divide_line_80 = np.percentile(active_time, 80)
divide_line_20 = np.percentile(active_time, 20)
divide_line_50 = np.percentile(active_time, 50)
print('divide line', divide_line_80, divide_line_20, divide_line_50)

# user history
labeled_behaviors = {}
for uid, uinfo in tqdm(filter_user.items(), total=len(filter_user), desc='relabel'):

    personal_times = []
    all_his = uinfo['his'] + uinfo['train'] + uinfo['valid'] + uinfo['test']
    for _, at in all_his:
        personal_times.append(at)
    personal_times = sorted(personal_times)
    personal_line_80 = np.percentile(personal_times, 80)
    personal_line_20 = np.percentile(personal_times, 20)
    personal_line_50 = np.percentile(personal_times, 50)
    uinfo['line'] = personal_line_80

    uidx = uinfo['uidx']
    labeled_behaviors[uidx] = {}
    labeled_behaviors[uidx] = {'pos': [], 'neg': [], 'cnt': 0, 'graph': []}

    labeled_behaviors[uidx]['cnt'] = len(uinfo['his'])
    assert(labeled_behaviors[uidx]['cnt'] > 0)
    for nid, read_seconds in uinfo['his']:

        if read_seconds > personal_line_80:
            labeled_behaviors[uidx]['pos'].append(news_dict[nid]['idx'])
        else:
            labeled_behaviors[uidx]['neg'].append(news_dict[nid]['idx'])

        if read_seconds > personal_line_20:
            labeled_behaviors[uidx]['graph'].append(news_dict[nid]['idx'])

# neighbor dict
for uid, uinfo in tqdm(filter_user.items(), total=len(filter_user), desc='find neighbor'):

    his_len = len(uinfo['his'])
    for i in range(his_len - 1):
        cur_node = news_dict[uinfo['his'][i][0]]['idx']
        for j in range(i + 1, his_len):
            n_node = news_dict[uinfo['his'][j][0]]['idx']
            neighbor_dict[cur_node].add(n_node)
            neighbor_dict[n_node].add(cur_node)

# simplified set
all_news_set = set(news_simplified_dict.keys())
for uid, uinfo in tqdm(filter_user.items(), total=len(filter_user), desc='sample neg'):

    pos_set = set()
    for nid, read_seconds in uinfo['train'] + uinfo['valid']:
        pos_set.add(news_dict[nid]['idx'])
        for neighbor_nid in neighbor_dict[news_dict[nid]['idx']]:
             pos_set.add(neighbor_nid)

    train_len = len(uinfo['train'])
    valid_len = len(uinfo['valid'])

    neg_len = train_len + valid_len
    remain_set = all_news_set - pos_set
    sampled = random.sample(remain_set, neg_len)
    uinfo['train_neg'] = sampled[:train_len]
    uinfo['valid_neg'] = sampled[train_len:]

# train tsv
impression_id = 1
train_arr = []
for uid, uinfo in tqdm(filter_user.items(), total=len(filter_user), desc='build train'):

    if len(uinfo['train']) <= 0:
        continue
    
    impre = []
    for nid, read_seconds in uinfo['train']:
        impre.append(news_dict[nid]['idx'] + '-1')
    for simplified_nid in uinfo['train_neg']:
        impre.append(simplified_nid + '-0')

    impre_str = ' '.join(impre)

    new_row = []
    new_row.append(impression_id)
    new_row.append(uinfo['uidx'])
    new_row.append(impre_str)
    train_arr.append(new_row)
    impression_id += 1

train_df = pd.DataFrame(train_arr)

# valid tsv
valid_impression_id = 1
valid_arr = []

for uid, uinfo in tqdm(filter_user.items(), total=len(filter_user), desc='build valid'):

    if len(uinfo['valid']) <= 0:
        continue
    
    impre = []
    for nid, read_seconds in uinfo['valid']:
        impre.append(news_dict[nid]['idx'] + '-1')
    for simplified_nid in uinfo['valid_neg']:
        impre.append(simplified_nid + '-0')

    impre_str = ' '.join(impre)

    new_row = []
    new_row.append(valid_impression_id)
    new_row.append(uinfo['uidx'])
    new_row.append(impre_str)
    valid_arr.append(new_row)
    valid_impression_id += 1

valid_df = pd.DataFrame(valid_arr)

# pickle.dump(filter_user, open('adressa/user_behaviors.pkl', 'wb'))
# json.dump(news_dict, open('adressa/news_dict.json', 'w'))
json.dump(labeled_behaviors, open('adressa/his_behaviors.json', 'w'))
train_df.to_csv('adressa/train_behaviors.tsv', index=None, header=None, sep='\t')
valid_df.to_csv('adressa/dev_behaviors.tsv', index=None, header=None, sep='\t')
json.dump(news_simplified_dict, open('adressa/news_dict.json', 'w'))
pickle.dump(neighbor_dict, open('adressa/neighbor.pkl', 'wb'))