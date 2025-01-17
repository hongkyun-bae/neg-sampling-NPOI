
run_name = "RNS"

##########NDCG
import math
def get_NDCG(groundtruth, pred_rank_list, k):
    count = 0
    dcg = 0
    for pred in pred_rank_list:
        if count >= k:
            break
        if groundtruth[pred] == 1:
            dcg += (1) / math.log2(count + 1 + 1)
        count += 1
    idcg = 0
    num_real_item = np.sum(groundtruth)
    num_item = int(min(num_real_item, k))
    for i in range(num_item):
        idcg += (1) / math.log2(i + 1 + 1)
    ndcg = dcg / idcg
    return ndcg


import time

import sys


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("./testsave/log_" + run_name + ".txt", "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger()

###############


import tensorflow as tf
from model.RankPOI import HParams, RankPOI
from batch_reader_rankpoi import Batcher
import numpy as np
import os, sys
import pandas as pd
import pickle
from tensorflow.python import pywrap_tensorflow

tf.flags.DEFINE_string("train_or_test", "train", "Value can be selected by train or test")
tf.flags.DEFINE_string("data_type", "TKY",
                       "the type of dataset, TKY or TKY")
FLAGS = tf.flags.FLAGS

train_data_path="../../data/Foursquare/{}/{}_PROCESS_DELETE_SPLIT.txt".format(FLAGS.data_type,FLAGS.data_type)
valid_data_path="../../data/Foursquare/{}/{}_VALID_SPLIT.csv".format(FLAGS.data_type,FLAGS.data_type)
test_data_path="../../data/Foursquare/{}/{}_TEST_SPLIT.csv".format(FLAGS.data_type,FLAGS.data_type)

poi_candidate_path = "../../data/Foursquare/{}/{}_FILTER_POI.txt".format(FLAGS.data_type,FLAGS.data_type)

alldata_path="../../data/Foursquare/{}/{}_PROCESS_DELETE.csv".format(FLAGS.data_type,FLAGS.data_type)

pre_model_path="./save/CategoryModel/{}/model.ckpt-25".format(FLAGS.data_type)
category_embedding_name="Model/CategoryModel/embedding/category_embedding"


# batcher_train=None
# batcher_valid=None
# batcher_test=None

save_dir = "./save/RANKPOI/{}".format(FLAGS.data_type)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# save_dir_best = os.path.join(save_dir, "best")
from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import minmax_scale
def sklearn_haversine(lat, lon):
    haversine = DistanceMetric.get_metric('haversine')
    latlon = np.hstack((np.radians(lat[:, np.newaxis]), np.radians(lon[:, np.newaxis])))
    dists = haversine.pairwise(latlon)

    a = [(1 - minmax_scale(i)).tolist() for i in dists]

    return np.array(a)


poi = pd.read_csv('../../data/Foursquare/TKY/TKY_VENUE_CAT_LON_LAT.csv')
# latlon = pd.DataFrame(poi).iloc[:, 1:].to_numpy()
lat, lon = pd.DataFrame(poi).iloc[:, 2:3].values.tolist(), pd.DataFrame(poi).iloc[:, 3:].values.tolist()


place_correlation = sklearn_haversine(np.squeeze(lat), np.squeeze(lon))
np.fill_diagonal(place_correlation,0)

print(place_correlation)
print(len(place_correlation))

def _train(hps, batcher):
    start_global_step = 1
    with tf.Graph().as_default(), tf.Session() as sess:
        with tf.variable_scope("Model_Train"):

            # reader = pywrap_tensorflow.NewCheckpointReader(pre_model_path)
            # pre_category_embedding=reader.get_tensor(category_embedding_name)
            train_model = RankPOI(hps)
            train_model.build_graph()
            # writer = tf.summary.FileWriter("./graph", sess.graph)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables(),max_to_keep=0)
            if os.path.exists(os.path.join(save_dir, "checkpoint")):
                print("continue to train")
                ckpt = tf.train.get_checkpoint_state(save_dir)
                saver.restore(sess, ckpt.model_checkpoint_path)
                print(ckpt.model_checkpoint_path)
                start_global_step = int(ckpt.model_checkpoint_path.split("-")[-1]) + 1

            print("start to train")
            for step in range(start_global_step, 200):
                losses = 0
                count = 0
                all_pre_5 = float(0)
                all_rec_5 = float(0)
                all_pre_10 = float(0)
                all_rec_10 = float(0)
                all_pre_15 = float(0)
                all_rec_15 = float(0)
                all_pre_20 = float(0)
                all_rec_20 = float(0)
                all_n_5=float(0)
                all_n_10=float(0)
                all_n_15=float(0)
                all_n_20=float(0)


                train_model.assign_global_step(sess, step)
                for user in np.random.permutation(hps.nb_users):
                    state = sess.run(train_model.initial_state)
                    batch_user, enc_poi, enc_neg_poi, enc_cat, enc_time, enc_windows, real_lenth,poi_category,label,poi_candidate,is_in_test= batcher.nextBatch_POI_Train(user)
                    if (enc_poi is not None):
                        # preference_cat = np.array(preference_cat_dict[user][0])
                        loss, poi_rank, _ = train_model.run_train_step(
                            sess, batch_user, enc_poi, enc_neg_poi, enc_cat, enc_time, enc_windows, real_lenth,
                            poi_category, state,place_correlation)
                        if is_in_test == 1:
                            pre_5, rec_5 = compute_pre_rec_poi(poi_rank, label, 5)
                            pre_10, rec_10 = compute_pre_rec_poi(poi_rank, label, 10)
                            pre_15, rec_15 = compute_pre_rec_poi(poi_rank, label, 15)
                            pre_20, rec_20 = compute_pre_rec_poi(poi_rank, label, 20)
                            n_5 = get_NDCG(label[0], poi_rank[0], 5)
                            n_10 = get_NDCG(label[0], poi_rank[0], 10)
                            n_15 = get_NDCG(label[0], poi_rank[0], 15)
                            n_20 = get_NDCG(label[0], poi_rank[0], 20)

                            all_pre_5 += float(pre_5)
                            all_rec_5 += float(rec_5)
                            all_pre_10 += float(pre_10)
                            all_rec_10 += float(rec_10)
                            all_pre_15 += float(pre_15)
                            all_rec_15 += float(rec_15)
                            all_pre_20 += float(pre_20)
                            all_rec_20 += float(rec_20)
                            all_n_5 += float(n_5)
                            all_n_10 += float(n_10)
                            all_n_15 += float(n_15)
                            all_n_20 += float(n_20)
                            losses += loss
                            count += 1

                print("step:{}, train_loss: {:.5f}, train_pre5:{:.5f}, train_rec5:{:.5f}, train_pre10:{:.5f}, train_rec10:{:.5f}, train_pre15:{:.5f}, train_rec15:{:.5f}, train_pre20:{:.5f}, train_rec20:{:.5f}, ndcg5:{:.5f}, ndcg10:{:.5f}, ndcg15:{:.5f}, ndcg20:{:.5f}"
                            .format(step, losses / count, all_pre_5 / count, all_rec_5 / count, all_pre_10 / count,
                                    all_rec_10 / count, all_pre_15 / count, all_rec_15 / count, all_pre_20 / count, all_rec_20 / count
                                    , all_n_5 / count, all_n_10 / count, all_n_15 / count, all_n_20 / count))

                checkpoint_path = os.path.join(save_dir, "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=step)
                # _test(hps,batcher)
                hps = hps._replace(mode="train")

def _test(hps,batcher):

    hps = hps._replace(mode="test")
    ckpt = tf.train.get_checkpoint_state(save_dir)

    with tf.Graph().as_default(), tf.Session() as sess:
        with tf.variable_scope("Model_Train"),tf.device("/cuda:0"):

            test_model = RankPOI(hps)
            test_model.build_graph()
            # writer = tf.summary.FileWriter("./graph", sess.graph)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())
            # saver.restore(sess, ckpt.model_checkpoint_path) #the newest
            # saver.restore(sess, os.path.join(save_dir, "model.ckpt-{}".format(step))) #assign

            # print("start to test")
            losses = 0
            count=0
            # all_pre_5=float(0)
            # all_rec_5 = float(0)
            # all_pre_10=float(0)
            # all_rec_10 = float(0)
            # all_pre_15=float(0)
            # all_rec_15= float(0)
            # all_pre_20 = float(0)
            # all_rec_20 = float(0)
            # all_n_5 = float(0)
            # all_n_10 = float(0)
            # all_n_15 = float(0)
            # all_n_20 = float(0)

            can_all_pre_5=float(0)
            can_all_rec_5 = float(0)
            can_all_pre_10=float(0)
            can_all_rec_10 = float(0)
            can_all_pre_15=float(0)
            can_all_rec_15= float(0)
            can_all_pre_20 = float(0)
            can_all_rec_20 = float(0)
            can_all_n_5 = float(0)
            can_all_n_10 = float(0)
            can_all_n_15 = float(0)
            can_all_n_20 = float(0)

            for step in range (200): #for mode 'test'
                saver.restore(sess, os.path.join(save_dir, "model.ckpt-{}".format(step + 1)))
            # for step in range (1):
            #     saver.restore(sess, ckpt.model_checkpoint_path)  # the newest

                for user in np.random.permutation(hps.nb_users):

                    state = sess.run(test_model.initial_state)
                    batch_user, enc_poi, enc_neg_poi, enc_cat, enc_time, enc_windows, real_lenth, poi_category, label, poi_candidate, is_in_test = batcher.nextBatch_POI_Train(
                        user)
                    if (enc_poi is not None):

                        loss, poi_rank = test_model.run_test_step(
                            sess, batch_user, enc_poi, enc_neg_poi, enc_cat, enc_time, enc_windows, real_lenth,
                            poi_category, state,place_correlation)
                        if is_in_test == 1:
                            losses += loss
                            count += 1

                            poi_rank_candidate = [[val for val in poi_rank[0] if val in poi_candidate]]
                            n_5 = get_NDCG(label[0], poi_rank_candidate[0], 5)
                            n_10 = get_NDCG(label[0], poi_rank_candidate[0], 10)
                            n_15 = get_NDCG(label[0], poi_rank_candidate[0], 15)
                            n_20 = get_NDCG(label[0], poi_rank_candidate[0], 20)

                            pre_5, rec_5 = compute_pre_rec_poi(poi_rank_candidate, label, 5)
                            pre_10, rec_10 = compute_pre_rec_poi(poi_rank_candidate, label, 10)
                            pre_15, rec_15 = compute_pre_rec_poi(poi_rank_candidate, label, 15)
                            pre_20, rec_20 = compute_pre_rec_poi(poi_rank_candidate, label, 20)
                            can_all_pre_5 += float(pre_5)
                            can_all_rec_5 += float(rec_5)
                            can_all_pre_10 += float(pre_10)
                            can_all_rec_10 += float(rec_10)
                            can_all_pre_15 += float(pre_15)
                            can_all_rec_15 += float(rec_15)
                            can_all_pre_20 += float(pre_20)
                            can_all_rec_20 += float(rec_20)
                            can_all_n_5 += float(n_5)
                            can_all_n_10 += float(n_10)
                            can_all_n_15 += float(n_15)
                            can_all_n_20 += float(n_20)

                print(
                    "loss: {:.5f},pre5@:{:.5f}, rec@5:{:.5f}, pre@10:{:.5f}, rec@10:{:.5f}, pre@15:{:.5f}, rec@15:{:.5f}, pre20:{:.5f}, rec20:{:.5f}, ndcg5:{:.5f}, ndcg10:{:.5f}, ndcg15:{:.5f}, ndcg20:{:.5f}"
                        .format(losses / count, can_all_pre_5 / count, can_all_rec_5 / count,
                                can_all_pre_10 / count,
                                can_all_rec_10 / count, can_all_pre_15 / count, can_all_rec_15 / count,can_all_pre_20 / count, can_all_rec_20 / count
                                , can_all_n_5 / count, can_all_n_10 / count, can_all_n_15 / count, can_all_n_20 / count))


def compute_pre_topkcat_one(target_batch,topk_cat):
    all_pre = 0
    count_possitive_real = 0
    count_possitive_predict = 0
    for i, j in enumerate(target_batch):
        if j[1] == 1:
            count_possitive_real += 1
            if i in topk_cat:
                count_possitive_predict += 1

    all_pre += count_possitive_predict / count_possitive_real
    return format(float(all_pre),".5f")

def compute_pre_rec_one(poi_rank,clf_poi,label,top_k):
    topk_index=np.lexsort(-poi_rank.T)
    count=0
    for i in range(top_k):
        if i<len(poi_rank):
            # print("topk_index:",topk_index)
            # print("clf_poi:",clf_poi)
            if clf_poi[topk_index[i]] in label:
                count +=1
    pre=format(float(count) / float(top_k),".5f")
    recall=format(float(count) / float(len(label)),".5f")
    return pre,recall


def precision(indices, batch_y, k, count, delta_dist):
    precision = 0
    for i in range(indices.size(0)):
        sort = indices[i]
        if batch_y[i].long() in sort[:k]:
            precision += 1
    return precision / count


def compute_pre_rec_poi(topk_poi,target_batch,top_k):

    topk_poi=topk_poi[0][0:top_k]
    target_batch=target_batch[0]
    count=0
    for i in range (len(topk_poi)):
        if target_batch[topk_poi[i]]==1:
           count += 1
    pre=format(float(count) / float(top_k),".5f")
    rec=format(float(count) / float(np.sum(target_batch)),".5f")
    return pre,rec

def main(_):
    hps = HParams(
        nb_users=2293,
        nb_items=12740,
        nb_times=24,
        nb_categories=315,
        num_windows=12,
        topk_poi=5,
        batch_size=1,
        handset_timesteps=1,
        hidden_size=64,
        min_lr=0.0001,
        lr=0.001,
        activ_prelu=0.1,
        enc_timesteps=100,
        mode="train",
        max_grad_norm=9
    )

    # global enc_timesteps_valid, batcher_valid, batcher_test, enc_timesteps_test
    if FLAGS.train_or_test == "train":
        batcher_train = Batcher.from_path_with_time(train_data_path,valid_data_path, poi_candidate_path,alldata_path, hps)
        hps = batcher_train._hps
        _train(hps, batcher_train)

    if FLAGS.train_or_test == "test":
        batcher_test = Batcher.from_path_with_time(train_data_path,test_data_path, poi_candidate_path,alldata_path, hps)
        hps = batcher_test._hps
        _test(hps, batcher_test)

if __name__ == '__main__':
    tf.app.run()
