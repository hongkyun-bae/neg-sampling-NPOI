#####3

run_name = "bri-random"

import pandas as pd
import time
import datetime
import os
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.utils.data as Data
from torch.backends import cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pickle
import sys
import codecs

import preprocess_longshort as preprocess
import model_longshort as model

global_seed = 256
SEED = 0
torch.manual_seed(SEED)
import random

torch.cuda.manual_seed(SEED)
import scipy.sparse
# corr= scipy.sparse.load_npz('./corr/bri_place_correlation_50.npz')
from sklearn.preprocessing import minmax_scale


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def cal_ndcg(prob, label, k):
    # log_prob (N, L), label (N), batch_size [*M]
    prob = prob.cpu().detach().numpy()[0]
    minimum = min(prob)
    normalized_X = [x - minimum for x in prob]
    normalized_X = minmax_scale(normalized_X)
    poinum = prob.shape[0] + 1
    y_true = np.zeros(poinum)
    y_true[label] = 1
    nd = ndcg_score(y_true, normalized_X, k=k)

    return nd

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def cal_ndcg(prob, label, k,vocab_poi):
    # log_prob (N, L), label (N), batch_size [*M]
    nd=0
    for i in range(prob.shape[0]):
        prob_cpu = prob[i].cpu().detach().numpy()
        y_true = np.zeros(vocab_poi+1)
        y_true[label[i].cpu()] = 1
        nd += ndcg_score(y_true, prob_cpu, k=k)
    return nd

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

batch_size = 32
hidden_size = 128
num_layers = 1
num_epochs = 20
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

data = pd.DataFrame(pd.read_table("input/bri_small2.txt", header=1, encoding="latin-1"))
data.columns = ["userid", "venueid", "catid", "catname", "latitute", "longitude", "timezone", "time"]

print("start preprocess")
pre_data = preprocess.sliding_varlen(data, batch_size)
print("pre done")

with open("pre_data.txt", "rb") as f:
    pre_data = pickle.load(f)

with open("long_term.pk", "rb") as f:
    long_term = pickle.load(f)

with open("cat_candidate.pk", "rb") as f:
    cat_candi = pickle.load(f)

with open("history_uid_loc.pk", "rb") as f:
    history_uid_loc = pickle.load(f)


# with open('long_term_feature.pk','rb') as f:
# 	long_term_feature = pickle.load(f)
long_term_feature = [0]

cat_candi = torch.cat((torch.Tensor([0]), cat_candi))
cat_candi = cat_candi.long()

[vocab_poi, vocab_cat, vocab_user, len_train, len_test] = pre_data["size"]

loader_train = pre_data["loader_train"]
loader_test = pre_data["loader_test"]

print("train set size: ", len_train)
print("test set size: ", len_test)
print("vocab_poi: ", vocab_poi)
print("vocab_cat: ", vocab_cat)

print("Train the Model...")

Model = model.long_short(
    embed_user,
    embed_poi,
    embed_cat,
    embed_hour,
    embed_week,
    hidden_size,
    num_layers,
    vocab_poi + 1,
    vocab_cat + 1,
    vocab_user + 1,
    vocab_hour,
    long_term,
    cat_candi,
    # len(long_term_feature[0]),
)

######################Model load and retrain###################################
# checkpoint = torch.load('./checkpoint_file/checkpoint_bri-random-small/checkpoint_19.tar')
# Model.load_state_dict(checkpoint['state_dict'])
#
# Model = Model.cuda()
# Model.eval()
#
#######
userid_cursor = False
results_cursor = False

Model = Model.cuda()

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(Model.parameters(), lr)


def precision(indices, batch_y, k, count, delta_dist):
    precision = 0
    for i in range(indices.size(0)):
        sort = indices[i]
        if batch_y[i].long() in sort[:k]:
            precision += 1
    return precision / count


def MAP(indices, batch_y, k, count):
    sum_precs = 0
    for i in range(indices.size(0)):
        sort = indices[i]
        ranked_list = sort[:k]
        hists = 0
        for n in range(len(ranked_list)):
            if ranked_list[n].cpu().numpy() in batch_y[i].long().cpu().numpy():
                hists += 1
                sum_precs += hists / (n + 1)
    return sum_precs / count


def recall(indices, batch_y, k, count, delta_dist):
    recall_correct = 0
    for i in range(indices.size(0)):
        sort = indices[i]
        if batch_y[i].long() in sort[:k]:
            recall_correct += 1
    return recall_correct / count


for epoch in range(0,30):
    start = time.time()
    Model = Model.train()
    total_loss = 0.0

    precision_1 = 0
    precision_5 = 0
    precision_10 = 0
    precision_20 = 0
    precision_25 = 0
    precision_15=0
    precision_50 = 0
    ndcg_1, ndcg_5, ndcg_10, ndcg_15, ndcg_20 = 0, 0, 0, 0, 0

    MAP_1 = 0
    MAP_5 = 0
    MAP_10 = 0
    MAP_20 = 0
    MAP_25 = 0
    MAP_50 = 0

    userid_wrong_train = {}
    userid_wrong_test = {}
    results_train = []
    results_test = []

    for step, (batch_x, batch_x_cat, batch_y, hours, batch_userid, hour_pre, week_pre) in enumerate(loader_train):
        Model.zero_grad()
        users = batch_userid.cuda()
        hourids = Variable(hours.long()).cuda()

        batch_x, batch_x_cat, batch_y, hour_pre, week_pre = (
            Variable(batch_x).cuda(),
            Variable(batch_x_cat).cuda(),
            Variable(batch_y).cuda(),
            Variable(hour_pre.long()).cuda(),
            Variable(week_pre.long()).cuda(),
        )

        poi_candidate = list(range(vocab_poi + 1))
        poi_candi = Variable(torch.LongTensor(poi_candidate)).cuda()
        cat_candi = Variable(cat_candi).cuda()
        outputs = Model(
            batch_x, batch_x_cat, users, hourids, hour_pre, week_pre, poi_candi, cat_candi
        )
        outputs=torch.sigmoid(outputs)
        outputs2 = outputs[:, -1, :].clone().detach()
        n_ep = outputs.size()[1]

        loss = 0
        for i in range(batch_x.size(0)):
            for j in range(n_ep):
                global_seed += 1
                poi_cand = list(set(poi_candidate) - set(history_uid_loc[users[i].item()]['loc'].tolist()))
                random.seed(global_seed)
                ran = random.sample(poi_cand, 10)
                neg = outputs[i, j, ran]
                loss += torch.log(sum(torch.exp(neg))) - outputs[i, j, batch_y[i, j]]

        loss.backward()
        optimizer.step()

        total_loss += float(loss)

        batch_y2 = batch_y[:, -1]

        out_p, indices = torch.sort(outputs2, dim=1, descending=True)
        count = float(len_train)
        delta_dist = 0
        precision_1 += precision(indices, batch_y2, 1, count, delta_dist)
        precision_5 += precision(indices, batch_y2, 5, count, delta_dist)
        precision_10 += precision(indices, batch_y2, 10, count, delta_dist)
        precision_15 += precision(indices, batch_y2, 15, count, delta_dist)
        precision_20 += precision(indices, batch_y2, 20, count, delta_dist)
        precision_25 += precision(indices, batch_y2, 25, count, delta_dist)
        precision_50 += precision(indices, batch_y2, 50, count, delta_dist)
        ndcg_1 += cal_ndcg(outputs2, batch_y2, 1, vocab_poi)
        ndcg_5 += cal_ndcg(outputs2, batch_y2, 5, vocab_poi)
        ndcg_10 += cal_ndcg(outputs2, batch_y2, 10, vocab_poi)
        ndcg_15 += cal_ndcg(outputs2, batch_y2, 15, vocab_poi)
        ndcg_20 += cal_ndcg(outputs2, batch_y2, 20, vocab_poi)

        MAP_1 += MAP(indices, batch_y2, 1, count)
        MAP_5 += MAP(indices, batch_y2, 5, count)
        MAP_10 += MAP(indices, batch_y2, 10, count)
        MAP_20 += MAP(indices, batch_y2, 20, count)
        MAP_25 += MAP(indices, batch_y2, 25, count)
        MAP_50 += MAP(indices, batch_y2, 50, count)

    print(
        "val:",
        "epoch: [{}/{}]\t".format(epoch, num_epochs),
        "loss: {:.4f}\t".format(total_loss),
        "recall@1: {:.4f}\t".format(precision_1),
        "recall@5: {:.4f}\t".format(precision_5),
        "recall@10: {:.4f}\t".format(precision_10),
        "recall@15: {:.4f}\t".format(precision_15),
        "recall@20: {:.4f}\t".format(precision_20),
        "recall@25: {:.4f}\t".format(precision_25),
        "recall@50: {:.4f}\t".format(precision_50),
        "MAP@1: {:.4f}\t".format(MAP_1),
        "MAP@5: {:.4f}\t".format(MAP_5),
        "MAP@10: {:.4f}\t".format(MAP_10),
        "MAP@20: {:.4f}\t".format(MAP_20),
        "MAP@25: {:.4f}\t".format(MAP_25),
        "MAP@50: {:.4f}\t".format(MAP_50),
    "NDCG@1: {:.4f}\t".format(ndcg_1/count),
    "NDCG@5: {:.4f}\t".format(ndcg_5/count),
    "NDCG@10: {:.4f}\t".format(ndcg_10/count),
    "NDCG@15: {:.4f}\t".format(ndcg_15 / count),
    "NDCG@20: {:.4f}\t".format(ndcg_20/count)

    )

    savedir = "checkpoint_file/checkpoint_" + run_name
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    savename = savedir + "/checkpoint" + "_" + str(epoch) + ".tar"

    torch.save({"epoch": epoch + 1, "state_dict": Model.state_dict(), }, savename)

    if epoch % 1 == 0:

        Model = Model.eval()

        total_loss = 0.0

        precision_1 = 0
        precision_5 = 0
        precision_10 = 0
        precision_15 = 0
        precision_20 = 0
        precision_25 = 0
        precision_50 = 0

        MAP_1 = 0
        MAP_5 = 0
        MAP_10 = 0
        MAP_20 = 0
        MAP_25 = 0
        MAP_50 = 0
        ndcg_1, ndcg_5, ndcg_10, ndcg_15, ndcg_20 = 0, 0, 0, 0, 0
        for step, (batch_x, batch_x_cat, batch_y, hours, batch_userid, hour_pre, week_pre) in enumerate(loader_test):
            Model.zero_grad()
            hourids = hours.long()
            users = batch_userid

            batch_x, batch_x_cat, batch_y, hour_pre, week_pre = (
                Variable(batch_x).cuda(),
                Variable(batch_x_cat).cuda(),
                Variable(batch_y).cuda(),
                Variable(hour_pre.long()).cuda(),
                Variable(week_pre.long()).cuda(),
            )
            users = Variable(users).cuda()
            hourids = Variable(hourids).cuda()

            outputs = Model(
                batch_x, batch_x_cat, users, hourids, hour_pre, week_pre, poi_candi, cat_candi
            )
            outputs = torch.sigmoid(outputs)
            outputs2 = outputs[:, -1, :].clone().detach()
            loss = 0
            total_loss += float(loss)

            batch_y2 = batch_y[:, -1]

            weights_output = outputs2.data

            outputs2 = weights_output  # +weights_classify# + weights_comatrix +weights_hour_prob
            out_p, indices = torch.sort(outputs2, dim=1, descending=True)

            count = float(len_test)

            precision_1 += precision(indices, batch_y2, 1, count, delta_dist)
            precision_5 += precision(indices, batch_y2, 5, count, delta_dist)
            precision_10 += precision(indices, batch_y2, 10, count, delta_dist)
            precision_15 += precision(indices, batch_y2, 15, count, delta_dist)
            precision_20 += precision(indices, batch_y2, 20, count, delta_dist)
            precision_25 += precision(indices, batch_y2, 25, count, delta_dist)
            precision_50 += precision(indices, batch_y2, 50, count, delta_dist)
            ndcg_1 += cal_ndcg(outputs2, batch_y2, 1, vocab_poi)
            ndcg_5 += cal_ndcg(outputs2, batch_y2, 5, vocab_poi)
            ndcg_10 += cal_ndcg(outputs2, batch_y2, 10, vocab_poi)
            ndcg_15 += cal_ndcg(outputs2, batch_y2, 15, vocab_poi)
            ndcg_20 += cal_ndcg(outputs2, batch_y2, 20, vocab_poi)
            MAP_1 += MAP(indices, batch_y2, 1, count)
            MAP_5 += MAP(indices, batch_y2, 5, count)
            MAP_10 += MAP(indices, batch_y2, 10, count)
            MAP_20 += MAP(indices, batch_y2, 20, count)
            MAP_25 += MAP(indices, batch_y2, 25, count)
            MAP_50 += MAP(indices, batch_y2, 50, count)

        print(
            "test:",
            "loss: {:.4f}\t".format(total_loss),
            "recall@1: {:.4f}\t".format(precision_1),
            "recall@5: {:.4f}\t".format(precision_5),
            "recall@10: {:.4f}\t".format(precision_10),
            "recall@15: {:.4f}\t".format(precision_15),
            "recall@20: {:.4f}\t".format(precision_20),
            "recall@25: {:.4f}\t".format(precision_25),
            "recall@50: {:.4f}\t".format(precision_50),
            "MAP@1: {:.4f}\t".format(MAP_1),
            "MAP@5: {:.4f}\t".format(MAP_5),
            "MAP@10: {:.4f}\t".format(MAP_10),
            "MAP@20: {:.4f}\t".format(MAP_20),
            "MAP@25: {:.4f}\t".format(MAP_25),
            "MAP@50: {:.4f}\t".format(MAP_50),
            "NDCG@1: {:.4f}\t".format(ndcg_1 / count),
            "NDCG@5: {:.4f}\t".format(ndcg_5 / count),
            "NDCG@10: {:.4f}\t".format(ndcg_10 / count),
            "NDCG@15: {:.4f}\t".format(ndcg_15 / count),
            "NDCG@20: {:.4f}\t".format(ndcg_20 / count)
        )
