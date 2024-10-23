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

import pandas as pd

df1 = pd.read_csv('./data/raw_POIs.txt', sep='	')
df1.columns = ["venueid", "lat", "lon", "cate",'country']

import datetime

# # dataset_UbiComp2016_Bri_for_GeoSAN.txt 파일 읽기
df2 = pd.read_csv('./data/Brightkite_totalCheckins.txt', sep='	') # 구분자가 탭이라고 가정
df2.columns = ["userid", "time", "lat", "lon","venueid"]


df1['lat'] = df1['lat'].astype(float)
df1['lon'] = df1['lon'].astype(float)
df2['lat'] = df2['lat'].astype(float)
df2['lon'] = df2['lon'].astype(float)

def truncate_to_5_decimal_places(value):
    str_value = str(value)
    if '.' in str_value:
        return str_value[:str_value.index('.') + 5]
    return str_value

# 'lat'과 'lon' 열에 함수 적용
df1['lat'] = df1['lat'].apply(truncate_to_5_decimal_places)
df1['lon'] = df1['lon'].apply(truncate_to_5_decimal_places)

df2['lat'] = df2['lat'].apply(truncate_to_5_decimal_places)
df2['lon'] = df2['lon'].apply(truncate_to_5_decimal_places)


# d.csv의 lng 열과 dataset_UbiComp2016_Bri_for_GeoSAN.txt의 lon 열을 비교하고, lat 열도 비교
merged_df = pd.merge(df2, df1[['lon', 'lat', 'cate']],on=['lon', 'lat'], how='left')


# spot_categories 열을 맨 마지막에 추가
merged_df['venueid'] = merged_df['lat'].astype(str) + '_' + merged_df['lon'].astype(str)

# venueid를 숫자로 변환
merged_df['venueid'] = merged_df['venueid'].astype('category').cat.codes


# 결과 저장
merged_df.to_csv('./input/Bri_catdm_plspl_updated.txt', sep=',', index=False)

data = pd.DataFrame(pd.read_table("./input/Bri_catdm_plspl_updated.txt", header=1, sep=','))#Bri_catdm_plspl_updated
data.columns = ["userid", "time","latitute", "longitude", "venueid","catid"]
deep_time=[]
timestamp=[]
data = data.dropna(subset=['catid'])
for i in range(len(data)):
    try:
        times =data['time'].values[i]
        t = datetime.datetime.strptime(times, '%Y-%m-%dT%H:%M:%SZ')
        deepmove_time=t.strftime("%Y-%m-%d %H:%M:%S")
        deep_time.append(deepmove_time)
    except:
        print(times)
data['deepmove_time'] = deep_time
data['timestamp']=data['deepmove_time']
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.sort_values(by='userid', inplace=True, ascending=True)
data.sort_values(by='timestamp', inplace=True, ascending=True)


data['venueid'] = data['venueid'].rank(method='dense').values
data['venueid'] = data['venueid'].astype(int)
data['userid'] = data['userid'].rank(method='dense').values
data['userid'] = data['userid'].astype(int)
data.sort_values(by='userid', inplace=True, ascending=True)
data.sort_values(by='deepmove_time', inplace=True, ascending=True)


data = data.drop_duplicates()

#################################################################################

# while True:
#     bef_hash=len(data.venueid.unique())
#     bef_poi=len(data.userid.unique())
#     data= data.groupby(['userid']).filter(lambda x: len(x.venueid) > 9) # 20개 체크인 이상 방문한 유저
#     data= data.groupby(['venueid']).filter(lambda x: len(x) > 2) # 10개 체크인 이상 방문된 poi
#     data= data.groupby(['userid']).filter(lambda x: len(x) > 100) #
#     aft_hash=len(data.venueid.unique())
#     aft_poi=len(data.userid.unique())
#     if bef_hash==aft_hash and bef_poi==aft_poi: # 수렴할때까지
#         break
#
from datetime import datetime, timedelta
# data.sort_values(by=["userid", "deepmove_time"], inplace=True)

data.loc[data.groupby('userid', as_index=False,group_keys=False).apply(lambda x: x.iloc[:int(0.8*len(x))]).index, 'tr'] = int(0)
data.loc[data.groupby('userid', as_index=False,group_keys=False).apply(lambda x: x.iloc[int(0.8*len(x)):]).index, 'tr'] = int(1)
data['tr'] = data['tr'].astype(int)
data['venueid'] = data['venueid'].rank(method='dense').values
data['venueid'] = data['venueid'].astype(int)
data['userid'] = data['userid'].rank(method='dense').values
data['userid'] = data['userid'].astype(int)
data['catid'] = data['catid'].rank(method='dense').values
data['catid'] = data['catid'].astype(int)

data = data.dropna(subset=['catid'])
data = data.dropna(subset=['userid'])
data = data.dropna(subset=['venueid'])

data['deepmove_time'] = pd.to_datetime(data['deepmove_time'])
data["prev_venueid"] = data["venueid"].shift(1)
data["prev_deepmove_time"] = data["deepmove_time"].shift(1)

# 삭제할 행의 인덱스를 저장할 리스트
to_delete = []
data.reset_index(drop=True, inplace=True)  # 인덱스를 리셋함

# 삭제할 행의 인덱스를 저장할 리스트

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

data= data.groupby(['userid']).filter(lambda x: len(x) > 5) #


from datetime import timedelta
print(len(data.userid.unique()))
print(len(data.venueid.unique()))
print(data.catid.max())


data.to_csv('./input/Bri_plspl.txt', sep=',', index=False)

