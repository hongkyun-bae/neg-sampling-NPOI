import torch

from load import *
import time
import random
from torch import optim
import torch.utils.data as data
from tqdm import tqdm
from models import *
run_name = "Bri-DHNSDist-16" ##hihi
import sys
import math
from sklearn.preprocessing import minmax_scale
def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best

def precision(indices, batch_y, k):
    precision = 0
    for i in range(indices.size(0)):
        sort = indices[i]
        if batch_y[i].long() in sort[:k]:
            precision += 1
    return precision


def MAP(indices, batch_y,k):
    sum_precs = 0
    for i in range(indices.size(0)):
        sort = indices[i]
        ranked_list = sort[:k]
        hists = 0
        for n in range(len(ranked_list)):
            if ranked_list[n].cpu().numpy() in batch_y[i].long().cpu().numpy():
                hists += 1
                sum_precs += hists / (n + 1)
    return sum_precs

def cal_ndcg(prob, label,k):
    # log_prob (N, L), label (N), batch_size [*M]
    prob=prob.cpu().detach().numpy()[0]
    minimum = min(prob)
    normalized_X = [x - minimum for x in prob]
    normalized_X = minmax_scale(normalized_X)
    poinum=prob.shape[0]+1
    y_true = np.zeros(poinum)
    y_true[label]=1
    nd=ndcg_score(y_true, normalized_X, k=k)

    return nd

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


def calculate_acc(prob, label):
    # log_prob (N, L), label (N), batch_size [*M]
    acc_train = [0, 0, 0, 0]
    for i, k in enumerate([5, 10, 15, 20]):
        # topk_batch (N, k)
        _, topk_predict_batch = torch.topk(prob, k=k)
        for j, topk_predict in enumerate(to_npy(topk_predict_batch)):
            # topk_predict (k)
            if to_npy(label)[j] in topk_predict:
                acc_train[i] += 1

    return np.array(acc_train)

import pandas as pd
import numpy as np
from sklearn.neighbors import DistanceMetric
dist = DistanceMetric.get_metric('haversine')

from sklearn.preprocessing import minmax_scale
import pandas as pd
def sampling_prob(prob, label, num_neg,history):
    num_label, l_m = prob.shape[0], prob.shape[1] # prob (N, L)
    label = label.view(-1)  # label (N)
    init_label = np.linspace(0, num_label-1, num_label)  # (N), [0 -- num_label-1]
    init_prob = torch.zeros(size=(num_label, num_neg+len(label)))  # (N, num_neg+num_label)
    all_index=list(range(0, l_m)) #topk의 인덱스이지만 topk후에는 1더해야함

    not_his=list(set(all_index)-set([x-1 for x in history.cpu().detach().numpy()]))#exclude history /hisytory는 1부터임 label 계산 어떻게 하는지

    global global_seed
    global_seed += 1
    random.seed(global_seed)


    mm_place=1-minmax_scale(place_correlation[label.cpu().detach().numpy()[0]])
    # not_his=list(set(not_his)-set())
    sample_list = random.sample(not_his, 500)
    #minmax

    val, index = torch.topk(prob[0, sample_list], k=50, sorted=False)

    top20=[sample_list[i] for i in index.cpu().detach().numpy()]
    val, index = torch.topk(torch.mul(prob[0, top20] - min(prob[0, top20]),
              torch.Tensor(mm_place[top20]).to('cuda:0')), k=num_neg, sorted=False)


    ind=index.cpu().detach().numpy()
    for k in range(num_label):
        for i in range(num_neg + len(label)):
            if i < len(label):
                init_prob[k, i] = prob[k, label[i]]
            else:
                init_prob[k, i] = prob[0, top20[ind[i-len(label)]]]

    return torch.FloatTensor(init_prob), torch.LongTensor(init_label)  # (N, num_neg+num_label), (N)


class DataSet(data.Dataset):
    def __init__(self, traj, m1, v, label, length):
        # (NUM, M, 3), (NUM, M, M, 2), (L, L), (NUM, M), (NUM), (NUM)
        self.traj, self.mat1, self.vec, self.label, self.length = traj, m1, v, label, length

    def __getitem__(self, index):
        traj = self.traj[index].to(device)
        mats1 = self.mat1[index].to(device)
        vector = self.vec[index].to(device)
        label = self.label[index].to(device)
        length = self.length[index].to(device)
        return traj, mats1, vector, label, length

    def __len__(self):  # no use
        return len(self.traj)


class Trainer:
    def __init__(self, model, record):
        # load other parameters
        self.model = model.to(device)
        self.records = record
        self.start_epoch = record['epoch'][-1] if load else 1
        self.num_neg = 10
        self.interval = 1000
        self.batch_size = 1 # N = 1
        self.learning_rate = 3e-3
        self.num_epoch = 50
        self.threshold = np.mean(record['acc_valid'][-1]) if load else 0  # 0 if not update

        # (NUM, M, 3), (NUM, M, M, 2), (L, L), (NUM, M, M), (NUM, M), (NUM) i.e. [*M]
        self.traj, self.mat1, self.mat2s, self.mat2t, self.label, self.len = \
            trajs, mat1, mat2s, mat2t, labels, lens
        # nn.cross_entropy_loss counts target from 0 to C - 1, so we minus 1 here.
        self.dataset = DataSet(self.traj, self.mat1, self.mat2t, self.label-1, self.len)
        self.data_loader = data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)

    def train(self):
        # set optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=1)
        early=0
        for t in range(self.num_epoch):
            # settings or validation and test
            valid_size, test_size = 0, 0
            acc_valid, acc_test = [0, 0, 0, 0], [0, 0, 0, 0]
            valid_precision_5, valid_precision_10, valid_precision_15, valid_precision_20 = 0, 0, 0, 0
            valid_MAP_5, valid_MAP_15, valid_MAP_10, valid_MAP_20 = 0, 0, 0, 0
            valid_NDCG_5, valid_NDCG_10, valid_NDCG_15, valid_NDCG_20 = 0, 0, 0, 0

            test_precision_5, test_precision_10, test_precision_15, test_precision_20 = 0, 0, 0, 0
            test_MAP_5, test_MAP_10, test_MAP_15, test_MAP_20 = 0, 0, 0, 0
            test_NDCG_5, test_NDCG_10, test_NDCG_15, test_NDCG_20 = 0, 0, 0, 0
            bar = tqdm(total=part)
            for step, item in enumerate(self.data_loader):
                # get batch data, (N, M, 3), (N, M, M, 2), (N, M, M), (N, M), (N)
                person_input, person_m1, person_m2t, person_label, person_traj_len = item


                # first, try batch_size = 1 and mini_batch = 1

                input_mask = torch.zeros((self.batch_size, max_len, 3), dtype=torch.long).to(device)
                m1_mask = torch.zeros((self.batch_size, max_len, max_len, 2), dtype=torch.float32).to(device)
                for mask_len in range(1, person_traj_len[0]+1):  # from 1 -> len
                    # if mask_len != person_traj_len[0]:
                    #     continue
                    input_mask[:, :mask_len] = 1.
                    m1_mask[:, :mask_len, :mask_len] = 1.

                    train_input = person_input * input_mask
                    train_m1 = person_m1 * m1_mask
                    train_m2t = person_m2t[:, mask_len - 1]
                    train_label = person_label[:, mask_len - 1]  # (N)
                    train_len = torch.zeros(size=(self.batch_size,), dtype=torch.long).to(device) + mask_len

                    prob = self.model(train_input, train_m1, self.mat2s, train_m2t, train_len)  # (N, L)

                    if mask_len <= int(person_traj_len[0] *0.8):  # only training
                        # nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                        prob_sample, label_sample = sampling_prob(prob, train_label, self.num_neg,person_label[0])
                        loss_train = F.cross_entropy(prob_sample, label_sample)
                        loss_train.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()

                    elif int(person_traj_len[0] *0.8)<mask_len<=int(person_traj_len[0] *0.9):  # only validation
                        valid_size += person_input.shape[0]
                        # v_prob_sample, v_label_sample = sampling_prob(prob_valid, valid_label, self.num_neg)
                        # loss_valid += F.cross_entropy(v_prob_sample, v_label_sample, reduction='sum')
                        acc_valid += calculate_acc(prob, train_label)

                        valid_NDCG_5 += cal_ndcg(prob, train_label, 5)
                        valid_NDCG_10 += cal_ndcg(prob, train_label, 10)
                        valid_NDCG_15 += cal_ndcg(prob, train_label, 15)
                        valid_NDCG_20 += cal_ndcg(prob, train_label, 20)
                    elif mask_len > int(person_traj_len[0] *0.9):  # only test
                        test_size += person_input.shape[0]
                        # v_prob_sample, v_label_sample = sampling_prob(prob_valid, valid_label, self.num_neg)
                        # loss_valid += F.cross_entropy(v_prob_sample, v_label_sample, reduction='sum')
                        acc_test += calculate_acc(prob, train_label)

                        test_NDCG_5 += cal_ndcg(prob, train_label, 5)
                        test_NDCG_10 += cal_ndcg(prob, train_label, 10)
                        test_NDCG_15 += cal_ndcg(prob, train_label, 15)
                        test_NDCG_20 += cal_ndcg(prob, train_label, 20)

                bar.update(self.batch_size)
            bar.close()
            print(
                "epoch:{}\t".format(self.start_epoch + t),
                "valid_NDCG@5: {:.4f}\t".format(valid_NDCG_5 / valid_size),
                "valid_NDCG@10: {:.4f}\t".format(valid_NDCG_10 / valid_size),
                "valid_NDCG@15: {:.4f}\t".format(valid_NDCG_15 / valid_size),
                "valid_NDCG@20: {:.4f}\t".format(valid_NDCG_20 / valid_size)
            )

            print(
                "epoch:{}\t".format(self.start_epoch + t),
                "test_NDCG@5: {:.4f}\t".format(test_NDCG_5 / test_size),
                "test_NDCG@10: {:.4f}\t".format(test_NDCG_10 / test_size),
                "test_NDCG@15: {:.4f}\t".format(test_NDCG_15 / test_size),
                "test_NDCG@20: {:.4f}\t".format(test_NDCG_20 / test_size),
            )

            acc_valid = np.array(acc_valid) / valid_size
            print(' valid_acc:{}'.format(acc_valid))

            acc_test = np.array(acc_test) / test_size
            print('test_acc:{}'.format(acc_test))

            self.records['acc_valid'].append(acc_valid)
            self.records['acc_test'].append(acc_test)
            self.records['ndcg'].append([test_NDCG_5 / test_size,test_NDCG_10 / test_size,test_NDCG_15 / test_size,test_NDCG_20 / test_size])
            self.records['epoch'].append(self.start_epoch + t)
            early+=1
            if self.threshold < np.mean(acc_valid):
                self.threshold = np.mean(acc_valid)
                early=0
                print('----------save epoch:--------',self.start_epoch + t)
                # save the model
                torch.save({'state_dict': self.model.state_dict(),
                            'records': self.records,
                            'time': time.time() - start},
                           methodname + dname + '.pth')

            if early>10:
                break
from sklearn.neighbors import DistanceMetric
def sklearn_haversine(lat, lon):
    haversine = DistanceMetric.get_metric('haversine')
    latlon = np.hstack((np.radians(lat[:, np.newaxis]), np.radians(lon[:, np.newaxis])))
    dists = haversine.pairwise(latlon)
    return 6371 * dists

if __name__ == '__main__':
    # load data
    dname = 'Bri'

    part = 1600
    st=part-100 #hihi

    file = open('./data/' + dname + '_data.pkl', 'rb')
    file_data = joblib.load(file)
    # tensor(NUM, M, 3), np(NUM, M, M, 2), np(L, L), np(NUM, M, M), tensor(NUM, M), np(NUM)
    [trajs, mat1, mat2s, mat2t, labels, lens, u_max, l_max] = file_data
    mat1, mat2s, mat2t, lens = torch.FloatTensor(mat1), torch.FloatTensor(mat2s).to(device), \
                               torch.FloatTensor(mat2t), torch.LongTensor(lens)

    # the run speed is very flow due to the use of location matrix (also huge memory cost)
    # please use a partition of the data (recommended)
    poi = np.load('./data/' + dname + '_POI.npy')
    # latlon = pd.DataFrame(poi).iloc[:, 1:].to_numpy()
    lat, lon = pd.DataFrame(poi).iloc[:, 1:2].values.tolist(), pd.DataFrame(poi).iloc[:, 2:].values.tolist()
    np.squeeze(lat), np.squeeze(lon)
    place_correlation = sklearn_haversine(np.squeeze(lat), np.squeeze(lon))
    np.fill_diagonal(place_correlation, 0)

    nonzero={}

    for i in range(l_max):
        nonzero[i]=place_correlation[i].tolist()
    # print(nonzero)


    methodname=str(part)+'DHNSDist'
    trajs, mat1, mat2t, labels, lens = \
        trajs[st:part], mat1[st:part], mat2t[st:part], labels[st:part], lens[st:part]

    ex = mat1[:, :, :, 0].max(), mat1[:, :, :, 0].min(), mat1[:, :, :, 1].max(), mat1[:, :, :, 1].min()

    stan = Model(t_dim=hours+1, l_dim=l_max+1, u_dim=u_max+1, embed_dim=50, ex=ex, dropout=0)
    num_params = 0

    load = False

    if load:
        checkpoint = torch.load(dname +methodname+  '.pth')
        stan.load_state_dict(checkpoint['state_dict'])
        start = time.time() - checkpoint['time']
        records = checkpoint['records']

    else:
        records = {'epoch': [], 'acc_valid': [], 'acc_test': [], 'ndcg': []}
        start = time.time()

    trainer = Trainer(stan, records)
    trainer.train()
    # trainer.inference()

