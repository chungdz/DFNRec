import pandas as pd
import numpy as np
import json

train = pd.read_csv('./data/train/behaviors.tsv', sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
train['date'] = train['time'].apply(lambda x: x.split()[0])
his_day = train[(train['date'] != '11/14/2019') & (train['date'] != '11/13/2019')]
target_day = train[train['date'] == '11/14/2019']

his_day.drop(columns=['date']).to_csv('./data/train/his_behaviors.tsv', sep="\t", encoding="utf-8", header=None, index=None)
print('his ', his_day.shape)
target_day.drop(columns=['date']).to_csv('./data/train/target_behaviors.tsv', sep="\t", encoding="utf-8", header=None, index=None)
print('target ', target_day.shape)

