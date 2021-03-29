import json
import datetime
import os
import pickle
from tqdm import tqdm, trange
import numpy as np

def collect(user_dict, news_dict, path):

    f = open(path, 'r')
    doc = f.readlines()

    for i in range(0, len(doc)):
        event = json.loads(doc[i])
        if "profile" not in event:
            continue
        
        uid = event["userId"]
        nid = event["id"]
        
        if nid not in news_dict:
            news_dict[nid] = {}
            news_dict[nid]['title'] = event["title"]
        
        if "activeTime" not in event:
            continue
        
        if uid not in user_dict:
            user_dict[uid] = []
        user_dict[uid].append((nid, event["activeTime"]))

root = 'data'
his_user_dict = {}
news_dict = {}
# his
print('parse his')
begin = datetime.date(2017,1,1)
end = datetime.date(2017,3,10)
for i in trange((end - begin).days + 1):
    day = begin + datetime.timedelta(days=i)
    datapath = os.path.join(root, day.strftime('%Y%m%d'))
    collect(his_user_dict, news_dict, datapath)

# train
print('parse train')
begin = datetime.date(2017,3,11)
end = datetime.date(2017,3,24)
train_user_dict = {}
for i in trange((end - begin).days + 1):
    day = begin + datetime.timedelta(days=i)
    datapath = os.path.join(root, day.strftime('%Y%m%d'))
    collect(train_user_dict, news_dict, datapath)
# valid
print('parse valid')
begin = datetime.date(2017,3,25)
end = datetime.date(2017,3,26)
valid_user_dict = {}
for i in trange((end - begin).days + 1):
    day = begin + datetime.timedelta(days=i)
    datapath = os.path.join(root, day.strftime('%Y%m%d'))
    collect(valid_user_dict, news_dict, datapath)
# test
print('parse test')
begin = datetime.date(2017,3,27)
end = datetime.date(2017,3,31)
test_user_dict = {}
for i in trange((end - begin).days + 1):
    day = begin + datetime.timedelta(days=i)
    datapath = os.path.join(root, day.strftime('%Y%m%d'))
    collect(test_user_dict, news_dict, datapath)

# remove clicks lower than 10
uidx = 0
filter_user = {}
for uid, uinfo in his_user_dict.items():
    if len(uinfo) > 4:
        filter_user[uid] = {}
        filter_user[uid]['his'] = uinfo
        filter_user[uid]['train'] = []
        filter_user[uid]['valid'] = []
        filter_user[uid]['test'] = []
        filter_user[uid]['uidx'] = uidx
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
# filter test user
test_num = 0
for uid, uinfo in test_user_dict.items():
    if uid in filter_user:
        filter_user[uid]['test'] = uinfo
        test_num +=  len(uinfo)
print('train num: ', train_num, 'valid num: ', valid_num, 'test num: ', test_num)

# active time check
active_time = []
for uid, uinfo in tqdm(filter_user.items(), total=len(filter_user), desc='get active time'):

    all_his = uinfo['his'] + uinfo['train'] + uinfo['valid'] + uinfo['test']
    for nid, at in all_his:
        active_time.append(at)

active_time = sorted(active_time)
divide_line = np.percentile(active_time, 80)
print('divide line', divide_line)

# change label
labeled_behaviors = {}
for uid, uinfo in tqdm(filter_user.items(), total=len(filter_user), desc='relabel'):

    personal_times = []
    all_his = uinfo['his'] + uinfo['train'] + uinfo['valid'] + uinfo['test']
    for _, at in all_his:
        personal_times.append(at)
    personal_times = sorted(personal_times)
    personal_line = np.percentile(personal_times, 80)

    uidx = uinfo['uidx']
    labeled_behaviors[uidx] = {}
    labeled_behaviors[uidx]['his'] = {'pos': [], 'neg': [], 'cnt': 0}
    labeled_behaviors[uidx]['train'] = {'pos': [], 'neg': [], 'cnt': 0}
    labeled_behaviors[uidx]['valid'] = {'pos': [], 'neg': [], 'cnt': 0}
    labeled_behaviors[uidx]['test'] = {'pos': [], 'neg': [], 'cnt': 0}

    labeled_behaviors[uidx]['his']['cnt'] = len(uinfo['his'])
    assert(labeled_behaviors[uidx]['his']['cnt'] > 0)
    for nid, read_seconds in uinfo['his']:

        if read_seconds >= personal_line:
            labeled_behaviors[uidx]['his']['pos'].append(nid)
        else:
            labeled_behaviors[uidx]['his']['neg'].append(nid)
    
    labeled_behaviors[uidx]['train']['cnt'] = len(uinfo['train'])
    if labeled_behaviors[uidx]['train']['cnt'] > 0:

        for nid, read_seconds in uinfo['train']:

            if read_seconds >= personal_line:
                labeled_behaviors[uidx]['train']['pos'].append(nid)
            else:
                labeled_behaviors[uidx]['train']['neg'].append(nid)
    
    labeled_behaviors[uidx]['valid']['cnt'] = len(uinfo['valid'])
    if labeled_behaviors[uidx]['valid']['cnt'] > 0:

        for nid, read_seconds in uinfo['valid']:

            if read_seconds >= personal_line:
                labeled_behaviors[uidx]['valid']['pos'].append(nid)
            else:
                labeled_behaviors[uidx]['valid']['neg'].append(nid)

    labeled_behaviors[uidx]['test']['cnt'] = len(uinfo['test'])
    if labeled_behaviors[uidx]['test']['cnt'] > 0:

        for nid, read_seconds in uinfo['test']:

            if read_seconds >= personal_line:
                labeled_behaviors[uidx]['test']['pos'].append(nid)
            else:
                labeled_behaviors[uidx]['test']['neg'].append(nid)

pickle.dump(filter_user, open('adressa/user_behaviors.pkl', 'wb'))
json.dump(news_dict, open('adressa/news_dict.json', 'w'))
json.dump(labeled_behaviors, open('adressa/labeled_behaviors.json', 'w'))