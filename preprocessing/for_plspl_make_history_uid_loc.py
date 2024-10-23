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


def sliding_varlen(data, batch_size):
    def timedelta(time1, time2):
        t1 = datetime.datetime.strptime(str(time1), '%a %b %d %H:%M:%S %z %Y')
        t2 = datetime.datetime.strptime(str(time2), '%a %b %d %H:%M:%S %z %Y')
        delta = t1 - t2
        time_delta = datetime.timedelta(days=delta.days, seconds=delta.seconds).total_seconds()
        return time_delta / 3600

    def get_entropy(x):
        x_value_list = set([x[i] for i in range(x.shape[0])])
        ent = 0.0
        for x_value in x_value_list:
            p = float(x[x == x_value].shape[0]) / x.shape[0]
            logp = np.log2(p)
            ent -= p * logp
        return ent

    #################################################################################
    # 1、sort the raw data in chronological order
    timestamp = []
    hour = []
    day = []
    week = []
    hour_48 = []
    for i in range(len(data)):
        times = data['time'].values[i]
        timestamp.append(time.mktime(time.strptime(times, '%a %b %d %H:%M:%S %z %Y')))
        t = datetime.datetime.strptime(times, '%a %b %d %H:%M:%S %z %Y')
        year = int(t.strftime('%Y'))
        day_i = int(t.strftime('%j'))
        week_i = int(t.strftime('%w'))
        hour_i = int(t.strftime('%H'))
        hour_i_48 = hour_i
        if week_i == 0 or week_i == 6:
            hour_i_48 = hour_i + 24

        if year == 2013:
            day_i = day_i + 366
        day.append(day_i)

        hour.append(hour_i)
        hour_48.append(int(hour_i_48))
        week.append(week_i)

    data['timestamp'] = timestamp
    data['hour'] = hour
    data['day'] = day
    data['week'] = week
    data['hour_48'] = hour_48

    data.sort_values(by='timestamp', inplace=True, ascending=True)

    #################################################################################
    # 2、filter users and POIs

    '''
    thr_venue = 1
    thr_user = 20
    user_venue = data.loc[:,['userid','venueid']]
    #user_venue = user_venue.drop_duplicates()

    venue_count = user_venue['venueid'].value_counts()
    venue = venue_count[venue_count.values>thr_venue]
    venue_index =  venue.index
    data = data[data['venueid'].isin(venue_index)]
    user_venue = user_venue[user_venue['venueid'].isin(venue_index)]
    del venue_count,venue,venue_index

    #user_venue = user_venue.drop_duplicates()
    user_count = user_venue['userid'].value_counts()
    user = user_count[user_count.values>thr_user]
    user_index = user.index
    data = data[data['userid'].isin(user_index)]
    user_venue = user_venue[user_venue['userid'].isin(user_index)]
    del user_count,user,user_index

    user_venue = user_venue.drop_duplicates()
    user_count = user_venue['userid'].value_counts()
    user = user_count[user_count.values>1]
    user_index = user.index
    data = data[data['userid'].isin(user_index)]
    del user_count,user,user_index

    '''
    data['userid'] = data['userid'].rank(method='dense').values
    data['userid'] = data['userid'].astype(int)
    data['venueid'] = data['venueid'].rank(method='dense').values
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
    cat_candidate = torch.Tensor(poi_cat['catid'].values)

    # 3、split data into train set and test set.
    #    extract features of each session for classification

    vocab_size_poi = int(max(data['venueid'].values))
    vocab_size_cat = int(max(data['catid'].values))
    vocab_size_user = int(max(data['userid'].values))

    print('vocab_size_poi: ', vocab_size_poi)
    print('vocab_size_cat: ', vocab_size_cat)
    print('vocab_size_user: ', vocab_size_user)

    train_x = []
    train_x_cat = []
    train_y = []
    train_hour = []
    train_userid = []
    train_indexs = []

    # the hour and week to be predicted
    train_hour_pre = []
    train_week_pre = []

    test_x = []
    test_x_cat = []
    test_y = []
    test_hour = []
    test_userid = []
    test_indexs = []

    # the hour and week to be predicted
    test_hour_pre = []
    test_week_pre = []

    long_term = {}

    long_term_feature = []

    data_train = {}
    train_idx = {}
    data_test = {}
    test_idx = {}

    data_train['datainfo'] = {'size_poi': vocab_size_poi + 1, 'size_cat': vocab_size_cat + 1,
                              'size_user': vocab_size_user + 1}

    len_session = 3
    user_lastid = {}
    #################################################################################
    # split data

    for uid, group in data.groupby('userid'):
        data_train[uid] = {}
        data_test[uid] = {}
        user_lastid[uid] = []
        inds_u = group.index.values
        split_ind = int(np.floor(0.8 * len(inds_u)))
        train_inds = inds_u[:split_ind]
        test_inds = inds_u[split_ind:]

        # get the features of POIs for user uid
        # long_term_feature.append(get_features(group.loc[train_inds]))

        long_term[uid] = {}
        '''
        long_term[uid]['loc'] = []
        long_term[uid]['hour'] = []
        long_term[uid]['week'] = []
        long_term[uid]['category'] = []

        lt_data = group.loc[train_inds]
        long_term[uid]['loc'].append(lt_data['venueid'].values)
        long_term[uid]['hour'].append(lt_data['hour'].values)
        long_term[uid]['week'].append(lt_data['week'].values)
        long_term[uid]['category'].append(lt_data['catid'].values)
        '''
        lt_data = group.loc[train_inds]
        long_term[uid]['loc'] = torch.LongTensor(lt_data['venueid'].values).cuda()

    with open('./history_uid_loc.pk', 'wb') as f:
        pickle.dump(long_term, f)
    data[["userid", "venueid", "catid", "catname", "latitute", "longitude", "timezone", "time"]].to_csv('nyc_small.txt',sep='\t',header=False,index=False)
# ---------------------------------------------------------------------------------------------------------
# long-term using attention

import pandas as pd
import time
import datetime
import os
import numpy as np

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

run_name = "test-ran-NYC"
#
# log = open("./testsave/log_" + run_name + ".txt", "w")
# sys.stdout = log
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training Parameters
batch_size = 32
hidden_size = 128
num_layers = 1
num_epochs = 25
lr = 0.001

vocab_hour = 24
vocab_week = 7

embed_poi = 300
embed_cat = 100
embed_user = 50
embed_hour = 20
embed_week = 7

print("emb_poi :", embed_poi)
print("emb_user :", embed_user)
print("hidden_size :", hidden_size)
print("lr :", lr)

data = pd.DataFrame(pd.read_table("./input/nyc_small.txt", header=None, encoding="latin-1"))
data.columns = ["userid", "venueid", "catid", "catname", "latitute", "longitude", "timezone", "time"]

print("start preprocess")
pre_data = sliding_varlen(data, batch_size)