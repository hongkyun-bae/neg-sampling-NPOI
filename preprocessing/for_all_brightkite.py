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


data = pd.DataFrame(pd.read_table("./data/Brightkite_totalCheckins.txt", header=None, encoding="latin-1"))
data.columns = ["userid", "time","latitute", "longitude", "venueid"]
data =data.dropna()
data = data[data['latitute'] != 0]
data = data[data['longitude'] != 0]
timestamp = []
hour = []
day = []
week = []
hour_48 = []
deep_time=[]
for i in range(len(data)):
    try:
        times =data['time'].values[i]
        t = datetime.datetime.strptime(times, '%Y-%m-%dT%H:%M:%SZ')
        deepmove_time=t.strftime("%Y-%m-%d %H:%M:%S")
        deep_time.append(deepmove_time)
    except:
        print(times)
data['deepmove_time'] = deep_time

#################################################################################
# 2、filter users and POIs
data['venueid'] = data['venueid'].rank(method='dense').values
data['venueid'] = data['venueid'].astype(int)
data['userid'] = data['userid'].rank(method='dense').values
data['userid'] = data['userid'].astype(int)
data.sort_values(by='userid', inplace=True, ascending=True)
data.sort_values(by='deepmove_time', inplace=True, ascending=True)


data = data.drop_duplicates()

#################################################################################

while True:
    bef_hash=len(data.venueid.unique())
    bef_poi=len(data.userid.unique())
    data= data.groupby(['userid']).filter(lambda x: len(x.venueid) > 19) # 20개 체크인 이상 방문한 유저
    data= data.groupby(['venueid']).filter(lambda x: len(x) > 9) # 10개 체크인 이상 방문된 poi
    data= data.groupby(['userid']).filter(lambda x: len(x) > 200) #
    aft_hash=len(data.venueid.unique())
    aft_poi=len(data.userid.unique())
    if bef_hash==aft_hash and bef_poi==aft_poi: # 수렴할때까지
        break
#
from datetime import datetime, timedelta
# data.sort_values(by=["userid", "deepmove_time"], inplace=True)
# data[["userid", "venueid","latitute", "longitude", "deepmove_time"]].to_csv('dataset_UbiComp2016_Bri_for_GeoSAN3.txt',sep=',',header=False,index=False)
data = pd.DataFrame(pd.read_table("dataset_UbiComp2016_Bri_for_GeoSAN4.txt",sep=',', header=None, encoding="latin-1"))
data.columns = ["userid", "venueid","latitute", "longitude", "deepmove_time"]
# # 중복된 행을 제외하고 차이 계산에 활용할 열 추가
data["deepmove_time"] = pd.to_datetime(data["deepmove_time"], format='%Y-%m-%d %H:%M:%S')

data["prev_venueid"] = data["venueid"].shift(1)
data["prev_deepmove_time"] = data["deepmove_time"].shift(1)

# 삭제할 행의 인덱스를 저장할 리스트
to_delete = []

# 연속된 행에서 두 번째 행을 삭제하는 조건 확인
for index, row in data.iterrows():
    if index ==0:
        continue
    if row["userid"] != data.at[index - 1, "userid"] or row["venueid"] != data.at[index - 1, "venueid"]:
        continue

    time_diff = row["deepmove_time"] - data.at[index - 1, "prev_deepmove_time"]
    if time_diff <= timedelta(days=1):
        to_delete.append(index - 1)

# 삭제할 행 제거
data.drop(to_delete, inplace=True)



# df_sorted = data.groupby('userid').apply(lambda x: x.sort_values('deepmove_time'))
#
# # 중복된 'venueid' 제거하여 이전 행만 남기기
# df_filtered = df_sorted[df_sorted['venueid'] != df_sorted['venueid'].shift(1)]

# data[['week','userid','week','week','deepmove_time','week','week','week','venueid']].to_csv('nyc.txt',sep='',header=False,index=False)
# data[['userid', 'deepmove_time', 'latitute', 'longitude', 'venueid']].to_csv('g_nyc.txt',sep='\t',header=False,index=False)
data['venueid'] = data['venueid'].rank(method='dense').values
data['venueid'] = data['venueid'].astype(int)
data['userid'] = data['userid'].rank(method='dense').values
data['userid'] = data['userid'].astype(int)
data.sort_values(by=['deepmove_time'], inplace=True, ascending=True)
# data.sort_values(by='deepmove_time', inplace=True, ascending=True)
#

data[["userid", "venueid","latitute", "longitude", "deepmove_time"]].to_csv('dataset_UbiComp2016_Bri_for_GeoSAN5.txt',sep=',',header=False,index=False)


# data[['use_ID', 'time', 'latitude', 'longitude', 'ite_ID']].to_csv('shan_tky.csv',sep='\t',header=False,index=False)
#user, time(2010-10-19T23:55:27Z), lat, lng, loc


print(len(data['userid'].unique()))
print(len(data['venueid'].unique()))
print(len(data))

