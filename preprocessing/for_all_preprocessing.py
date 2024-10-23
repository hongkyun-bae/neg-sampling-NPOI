import pandas as pd
import numpy as np



# _, uid, _, _, tim, _, _, tweet, pid = line.strip('\r\n').split('')

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import datetime
import pickle
import time
import os




###Small 개수
# user 5개 미만 제거
# item 5개 미만 제거


data = pd.DataFrame(pd.read_table("./data/dataset_TSMC2014_TKY.txt", header=None, encoding="latin-1"))
data.columns = ["userid", "venueid", "catid", "catname", "latitute", "longitude", "timezone", "time"]


timestamp = []
hour = []
day = []
week = []
hour_48 = []
deep_time=[]
for i in range(len(data)):
    times = data['time'].values[i]
    timestamp.append(time.mktime(time.strptime(times, '%a %b %d %H:%M:%S %z %Y')))
    t = datetime.datetime.strptime(times, '%a %b %d %H:%M:%S %z %Y')
    year = int(t.strftime('%Y'))
    day_i = int(t.strftime('%j'))
    week_i = int(t.strftime('%w'))
    hour_i = int(t.strftime('%H'))
    deepmove_time=t.strftime("%Y-%m-%dT%H:%M:%SZ")
    hour_i_48 = hour_i
    if week_i == 0 or week_i == 6:
        hour_i_48 = hour_i + 24

    if year == 2013:
        day_i = day_i + 366
    day.append(day_i)

    hour.append(hour_i)
    hour_48.append(int(hour_i_48))
    week.append(week_i)
    deep_time.append(deepmove_time)
data['deepmove_time'] = deep_time
data['timestamp'] = timestamp
data['hour'] = hour
data['day'] = day
data['week'] = week
data['hour_48'] = hour_48



#################################################################################
# 2、filter users and POIs
data['venueid'] = data['venueid'].rank(method='dense').values
data['venueid'] = data['venueid'].astype(int)
data['userid'] = data['userid'].rank(method='dense').values
data['userid'] = data['userid'].astype(int)
for venueid, group in data.groupby('venueid'):
    indexs = group.index
    if len(set(group['catid'].values)) > 1:
        for i in range(len(group)):
            data.loc[indexs[i], 'catid'] = group.loc[indexs[0]]['catid']


data = data.drop_duplicates()
data['catid'] = data['catid'].rank(method='dense').values

#################################################################################
poi_cat = data[['venueid', 'catid']]
poi_cat = poi_cat.drop_duplicates()
poi_cat = poi_cat.sort_values(by='venueid')

while True:
    bef_hash=len(data.venueid.unique())
    bef_poi=len(data.userid.unique())
    data= data.groupby(['userid']).filter(lambda x: len(x.venueid) > 5) # 7개 이상 poi 방문한 유저만
    data= data.groupby(['venueid']).filter(lambda x: len(x.userid) > 5) # 7명 이상 유저가 방문한 poi만
    aft_hash=len(data.venueid.unique())
    aft_poi=len(data.userid.unique())
    if bef_hash==aft_hash and bef_poi==aft_poi: # 수렴할때까지
        break

data.sort_values(by='userid', inplace=True, ascending=True)
data.sort_values(by='timestamp', inplace=True, ascending=True)

# data[['week','userid','week','week','deepmove_time','week','week','week','venueid']].to_csv('nyc.txt',sep='',header=False,index=False)
# data[['userid', 'deepmove_time', 'latitute', 'longitude', 'venueid']].to_csv('g_nyc.txt',sep='\t',header=False,index=False)
data[["userid", "venueid", "catid", "catname", "latitute", "longitude", "timezone", "time"]].to_csv('tky_small.txt',sep='\t',header=False,index=False)


# data[['use_ID', 'time', 'latitude', 'longitude', 'ite_ID']].to_csv('shan_tky.csv',sep='\t',header=False,index=False)
#user, time(2010-10-19T23:55:27Z), lat, lng, loc


print(data.columns)
