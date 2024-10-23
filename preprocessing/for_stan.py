import numpy as np
import datetime
import time
import pandas as pd
from datetime import timedelta

def sliding_varlen(data):

    timestamp = []
    hour = []
    day = []
    week = []
    hour_48 = []
    time_dt=[]
    for i in range(len(data)):
        times = data['time'].values[i]
        timestamp.append(time.mktime(time.strptime(times, '%Y-%m-%d %H:%M:%S')))
        t = datetime.datetime.strptime(times, '%Y-%m-%d %H:%M:%S')
        time_dt.append(t)

    data['time_dt'] =time_dt
    data['timestamp'] = timestamp

    data.sort_values(by='timestamp', inplace=True, ascending=True)

    data['userid'] = data['userid'].rank(method='dense').values
    data['userid'] = data['userid'].astype(int)
    data['venueid'] = data['venueid'].rank(method='dense').values
    data['userid'] = data['userid'].astype(int)

    data_npy = data[['userid','venueid',  'latitute', 'longitude','timestamp','time_dt']]

    data_npy = data_npy.sort_values(by=['userid', 'timestamp'])
    fa = data_npy

    fa['venueid'] = fa.venueid.astype('category').cat.codes +1 # poiID 정수형으로 재설정
    fa['userid'] = fa.userid.astype('category').cat.codes  +1# hashvin 정수형으로 재설정
    fa = fa.sort_values(by=['userid', 'timestamp'])
    # fa.timestamp =

    fa['time_dt']=(fa['time_dt']-min(fa['time_dt'])).dt.total_seconds()/60

    # 1, 698, 99366
    check_ = fa[['userid', 'venueid', 'time_dt']]
    df_POI = fa[['venueid', 'latitute', 'longitude']]
    df_POI = df_POI.drop_duplicates(['venueid'])

    check_ = check_.astype(int)
    df_POI['venueid'] = df_POI['venueid'].astype(int)

    np.save('../STAN/data/Bri.npy', check_)
    np.save('../STAN/data/Bri_POI.npy', df_POI)


# for NYC, TKY data
# data = pd.DataFrame(pd.read_table("./input/dataset_TSMC2014_NYC.txt", header=None, encoding="latin-1"))

# for Brightkite data
data = pd.DataFrame(pd.read_table("./input/dataset_UbiComp2016_Bri_for_GeoSAN.txt",sep=',', header=None, encoding="latin-1"))

data.columns = ["userid", "venueid","latitute", "longitude","time"]

print("start preprocess")
pre_data = sliding_varlen(data)

