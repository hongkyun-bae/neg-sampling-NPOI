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

# import pandas as pd
#
# df1 = pd.read_csv('./data/raw_POIs.txt', sep='	')
# df1.columns = ["venueid", "lat", "lon", "cate",'country']
#
# import datetime
#
# # # dataset_UbiComp2016_Bri_for_GeoSAN.txt 파일 읽기
# df2 = pd.read_csv('./data/Brightkite_totalCheckins.txt', sep='	') # 구분자가 탭이라고 가정
# df2.columns = ["userid", "time", "lat", "lon","venueid"]
#
#
# df1['lat'] = df1['lat'].astype(float)
# df1['lon'] = df1['lon'].astype(float)
# df2['lat'] = df2['lat'].astype(float)
# df2['lon'] = df2['lon'].astype(float)
#
# def truncate_to_5_decimal_places(value):
#     str_value = str(value)
#     if '.' in str_value:
#         return str_value[:str_value.index('.') + 5]
#     return str_value
#
# # 'lat'과 'lon' 열에 함수 적용
# df1['lat'] = df1['lat'].apply(truncate_to_5_decimal_places)
# df1['lon'] = df1['lon'].apply(truncate_to_5_decimal_places)
#
# df2['lat'] = df2['lat'].apply(truncate_to_5_decimal_places)
# df2['lon'] = df2['lon'].apply(truncate_to_5_decimal_places)
#
#
# # d.csv의 lng 열과 dataset_UbiComp2016_Bri_for_GeoSAN.txt의 lon 열을 비교하고, lat 열도 비교
# merged_df = pd.merge(df2, df1[['lon', 'lat', 'cate']],on=['lon', 'lat'], how='left')
#
#
# # spot_categories 열을 맨 마지막에 추가
# merged_df['venueid'] = merged_df['lat'].astype(str) + '_' + merged_df['lon'].astype(str)
#
# # venueid를 숫자로 변환
# merged_df['venueid'] = merged_df['venueid'].astype('category').cat.codes
#
#
# # 결과 저장
# merged_df.to_csv('./input/Bri_catdm_plspl_updated.txt', sep=',', index=False)

data = pd.DataFrame(pd.read_table("./input/Bri_plspl.txt", header=1, sep=','))#Bri_catdm_plspl_updated
data.columns = ['userid',	'venueid'	,'catid',	'catname'	,'latitute',	'longitude'	,'timezone',	'time']
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

from datetime import timedelta
print(len(data.userid.unique()))
print(len(data.venueid.unique()))
print(data.catid.max())


# 'userid' 별 마지막 데이터 추출
last_data = data.drop_duplicates(subset='userid', keep='last')

# 마지막 데이터에서 이전 24시간 내의 행 필터링
filtered_data = []
for i in range(len(last_data)):
    user_id = last_data.iloc[i]['userid']
    last_time = last_data.iloc[i]['timestamp']
    filtered = data[(data['userid'] == user_id) & (data['timestamp'] >= last_time - timedelta(seconds=24 * 60 * 60))]
    filtered_data.append(filtered)

# 필터링된 데이터프레임에서 'tr' 열에 1 할당
for filtered in filtered_data:
    data.loc[filtered.index, 'tr'] = 1

data['tr'] = np.where(data['tr'] == 1, 1, 0)

#################################################################################

# 2、filter users and POIs
##Bri_TEST_SPLIT
##Bri_TRAIN_TEST_SPLIT
##Bri_VALID_SPLIT
##Bri_VALID_TEST_SPLIT

data[["userid", "venueid", "catid", "latitute", "longitude", "deepmove_time"]].to_csv('./input/Bri_PROCESS_DELETE.csv',sep=',',header=False,index=False)

train= data.groupby('userid', as_index=False,group_keys=False).apply(lambda x: x.iloc[:int(0.8*len(x))])
last_train = train.drop_duplicates(subset='userid', keep='last')

# 마지막 데이터에서 이전 24시간 내의 행 필터링
filtered_train = []
for i in range(len(last_train)):
    user_id = last_train.iloc[i]['userid']
    last_time = last_train.iloc[i]['timestamp']
    # filtered = train[(train['userid'] == user_id) & (train['timestamp'] >= last_time - 24*60*60)]
    filtered = train[(train['userid'] == user_id) & (train['timestamp'] >= last_time - timedelta(seconds=24 * 60 * 60))]
    filtered_train.append(filtered)

# 필터링된 데이터프레임에서 'tr' 열에 1 할당
for filtered in filtered_train:
    train.loc[filtered.index, 'train_tr'] = 1

train['train_tr'] = np.where(train['train_tr'] == 1, 1, 0)
train['train_tr'] = train['train_tr'].astype(int)

test= data.groupby('userid', as_index=False,group_keys=False).apply(lambda x: x.iloc[int(0.8*len(x)):])
last_test = test.drop_duplicates(subset='userid', keep='last')

# 마지막 데이터에서 이전 24시간 내의 행 필터링
filtered_test = []
for i in range(len(last_test)):
    user_id = last_test.iloc[i]['userid']
    last_time = last_test.iloc[i]['timestamp']
    filtered = test[(test['userid'] == user_id) & (test['timestamp'] >= last_time - timedelta(seconds=24 * 60 * 60))]

    filtered_test.append(filtered)

# 필터링된 데이터프레임에서 'tr' 열에 1 할당
for filtered in filtered_test:
    test.loc[filtered.index, 'train_tr'] = 1

test['train_tr'] = np.where(test['train_tr'] == 1, 1, 0)
test['train_tr'] = test['train_tr'].astype(int)

valid= data.groupby('userid', as_index=False,group_keys=False).apply(lambda x: x.iloc[int(0.7*len(x)):int(0.8*len(x))])
last_valid = valid.drop_duplicates(subset='userid', keep='last')

# 마지막 데이터에서 이전 24시간 내의 행 필터링
filtered_valid = []
for i in range(len(last_valid)):
    user_id = last_valid.iloc[i]['userid']
    last_time = last_valid.iloc[i]['timestamp']
    filtered = valid[(valid['userid'] == user_id) & (valid['timestamp'] >= last_time - timedelta(seconds=24 * 60 * 60))]
    filtered_valid.append(filtered)

# 필터링된 데이터프레임에서 'tr' 열에 1 할당
for filtered in filtered_valid:
    valid.loc[filtered.index, 'train_tr'] = 1
valid['train_tr'] = np.where(valid['train_tr'] == 1, 1, 0)
valid['train_tr'] = valid['train_tr'].astype(int)


valid_test= data.groupby('userid', as_index=False,group_keys=False).apply(lambda x: x.iloc[int(0.7*len(x)):])
last_valid_test = valid_test.drop_duplicates(subset='userid', keep='last')

# 마지막 데이터에서 이전 24시간 내의 행 필터링
filtered_valid_test = []
for i in range(len(last_valid_test)):
    user_id = last_valid_test.iloc[i]['userid']
    last_time = last_valid_test.iloc[i]['timestamp']
    filtered = valid_test[(valid_test['userid'] == user_id) & (valid_test['timestamp'] >= last_time - timedelta(seconds=24 * 60 * 60))]
    filtered_valid_test.append(filtered)

# 필터링된 데이터프레임에서 'tr' 열에 1 할당
for filtered in filtered_valid_test:
    valid_test.loc[filtered.index, 'train_tr'] = 1

valid_test['train_tr'] = np.where(valid_test['train_tr'] == 1, 1, 0)
valid_test['train_tr'] = valid_test['train_tr'].astype(int)


train[["userid", "venueid", "catid", "latitute", "longitude", "deepmove_time",'train_tr']].to_csv('./input/Bri_TRAIN_SPLIT.csv',sep=',',header=False,index=False)
valid[["userid", "venueid", "catid", "latitute", "longitude", "deepmove_time",'train_tr']].to_csv('./input/Bri_VALID_SPLIT.csv',sep=',',header=False,index=False)
valid_test[["userid", "venueid", "catid", "latitute", "longitude", "deepmove_time",'train_tr']].to_csv('./input/Bri_VALID_TEST_SPLIT.csv',sep=',',header=False,index=False)
test[["userid", "venueid", "catid", "latitute", "longitude", "deepmove_time",'train_tr']].to_csv('./input/Bri_TEST_SPLIT.csv',sep=',',header=False,index=False)


data[["userid", "venueid", "catid", "latitute", "longitude", "deepmove_time",'tr']].to_csv('./input/Bri_PROCESS_DELETE_SPLIT.txt',sep=',',header=False,index=False)

data=data.drop_duplicates('venueid')
data.sort_values(by='venueid', inplace=True, ascending=True)
data[["venueid", "catid", "latitute", "longitude"]].to_csv('./input/Bri_VENUE_CAT_LON_LAT.csv',sep=',',header=False,index=False)


# data[['use_ID', 'time', 'latitude', 'longitude', 'ite_ID']].to_csv('shan_Bri.csv',sep='\t',header=False,index=False)
#user, time(2010-10-19T23:55:27Z), lat, lng, loc

