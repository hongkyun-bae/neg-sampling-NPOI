import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch.nn.functional as F
from utils import serialize, unserialize, LadderSampler, generate_square_mask, reset_random_seed, generate_decoder_mask, \
    get_visited_locs
from quadkey_utils import collect_fn_quadkey,collect_fn_test, collect_fn_neg_quadkey_included
from model import QuadKeyLocPredictor
from dataset import NegInclLSBNDataset
from quadkey_dataset import QuadKeyLBSNDataset
from tqdm import tqdm
import neg_sampler
import joblib
from near_loc_query import LocQuerySystem
import loss
import time as Time
from collections import Counter, namedtuple
from utils import get_cosine_similarity

MarkovChainModel = namedtuple('MarkovChainModel', ['mc', 'states', "state2idx", 'count', 'visited_locs'])


def evaluate(model, test_dataset, negative_sampler, region_processer, loc2quadkey, device, batch_size=32, num_neg=100,
             neg_given=False):
    model.eval()
    # if neg_given:
    # loader = DataLoader(test_dataset, batch_size=batch_size,
    #                         collate_fn=lambda e: collect_fn_quadkey(e, test_dataset, region_processer,
    #                                                                              loc2quadkey))

    test_dataset = [src for src in test_dataset if src !=[]]  # 빈 리스트인 src 원소를 제거

    loader = DataLoader(test_dataset,
                              batch_size=batch_size,
                             collate_fn=lambda e: collect_fn_test(e, test_dataset, negative_sampler,
                                                                     region_processer, loc2quadkey, k=num_neg))

    # else:
    #     raise NotImplementedError("must ·provide eval sorting samples.")
    cnt = Counter()
    array = np.zeros(num_neg + 1)
    with torch.no_grad():
        for _, (user, loc, time, region, trg, trg_reg, _, _, ds) in enumerate(loader):
            user = user.to(device)
            loc = loc.to(device)
            time = time.to(device)
            region = region.to(device)
            trg = trg.to(device)
            trg_reg = trg_reg.to(device)
            src_mask = pad_sequence([torch.zeros(e, dtype=torch.bool).to(device) for e in ds], batch_first=True,
                                    padding_value=True)
            att_mask = generate_square_mask(max(ds), device)
            output = model(user, loc, region, time, att_mask, src_mask, trg, trg_reg, None, ds=ds)
            idx = output.sort(descending=True, dim=0)[1]
            order = idx.topk(1, dim=0, largest=False)[1]
            cnt.update(order.squeeze().tolist())

    for k, v in cnt.items():
        array[k] = v


    # hit rate and NDCG
    hr = array.cumsum()
    recall_at_5 = hr[5] / hr.max() if len(hr) > 5 else 0
    recall_at_10 = hr[10] / hr.max() if len(hr) > 10 else 0
    recall_at_15 = hr[15] / hr.max() if len(hr) > 15 else 0
    recall_at_20 = hr[20] / hr.max() if len(hr) > 20 else 0

    ndcg = 1 / np.log2(np.arange(0, num_neg + 1) + 2)
    ndcg = ndcg * array
    ndcg = ndcg.cumsum() / hr.max()
    hr = hr / hr.max()

    return hr[:10], ndcg[:20],recall_at_5,recall_at_10,recall_at_15,recall_at_20


def semantic_overlap(model, idx2loc, n):
    category_to_type = unserialize('./data/CategoryToType.json')
    model.to(device)
    category_embedding = model.state_dict()['emb_loc.lookup_table']
    category_embedding = np.array(category_embedding.cpu())

    mean_match_rate, category_num = 0, len(train_dataset.idx2loc.values())

    f = open('geosan@100_Bri.txt', 'w', encoding='utf-8')
    for i, category in enumerate(idx2loc.values()):
        f.write(category + ',')
        f.write(','.join(list(map(str, category_embedding[i]))) + '\n')

    f.close()

    for i in range(1, category_num):
        rank_list = {}
        current_embedding = category_embedding[i]
        for j in range(1, category_num):
            if i == j: continue
            candidate_embedding = category_embedding[j]
            cos_sim = get_cosine_similarity(current_embedding, candidate_embedding)
            rank_list.update({idx2loc[j]: cos_sim})
        rank_list = sorted(rank_list.items(), key=lambda x: x[1], reverse=False)
        category_rank = [t[0] for t in rank_list][-n:]
        match_count = 0
        for candidate_category in category_rank:
            if category_to_type.get(idx2loc[i]) == category_to_type.get(candidate_category):
                match_count += 1

        match_rate = match_count / n
        mean_match_rate += match_rate
    mean_match_rate = mean_match_rate / (category_num - 1)

    return mean_match_rate


def save_embed(model, idx2loc, n):
    category_embedding = model['emb_loc.lookup_table']
    category_embedding = np.array(category_embedding.cpu())
    f = open('geosan@100_Bri.txt', 'w', encoding='utf-8')
    for i, category in enumerate(idx2loc.values()):
        f.write(category + ',')
        f.write(','.join(list(map(str, category_embedding[i]))) + '\n')

    f.close()


def train(model, train_dataset, test_dataset, optimizer, loss_fn, negative_sampler, test_sampler, region_processer,
          loc2quadkey, device, num_neg=5, batch_size=32,
          num_epochs=10, writer=None, save_path=None, batch_size_test=32, num_neg_test=100, test_neg_given=False,
          num_workers=5, save_results='data/result.txt'):

    for epoch_idx in range(num_epochs):
        start_time = Time.time()
        running_loss = 0.
        processed_batch = 0.
        data_loader = DataLoader(train_dataset, sampler=LadderSampler(train_dataset, batch_size),
                             num_workers=num_workers, batch_size=batch_size,
                             collate_fn=lambda e: collect_fn_quadkey(e, train_dataset, negative_sampler,
                                                                         region_processer, loc2quadkey, k=num_neg))

        # num_batch = len(data_loader)
        # print("=====epoch {:>2d}=====".format(epoch_idx + 1))
        Time.sleep(0.1)
        batch_iterator = tqdm(enumerate(data_loader), total=len(data_loader), leave=True)

        model.train()
        for batch_idx, (user, loc, time, region, trg, trg_reg, trg_nov, sample_probs, ds) in batch_iterator:
            user = user.to(device)
            loc = loc.to(device)
            time = time.to(device)
            region = region.to(device)
            trg = trg.to(device)
            trg_reg = trg_reg.to(device)
            sample_probs = sample_probs.to(device)
            optimizer.zero_grad()
            src_mask = pad_sequence([torch.zeros(e, dtype=torch.bool).to(device) for e in ds], batch_first=True,
                                    padding_value=True)
            att_mask = generate_square_mask(max(ds), device)
            output = model(user, loc, region, time, att_mask, src_mask, trg, trg_reg, att_mask.repeat(num_neg + 1, 1))

            output = output.view(-1, loc.size(0), loc.size(1)).permute(2, 1, 0)

            pos_score, neg500_score = output.split([1, 10], -1) # 디엔에스는 500으로 하면 되고, 랜덤 는 10으로 하면됨

            # neg_score,ind=torch.topk(neg500_score,10,sorted=False)

            # pos_score, neg_score = output.split([1, num_neg], -1)
            # torch.ones(len(trg_seq), 10, dtype=torch.float32)
            loss = loss_fn(pos_score, neg500_score, sample_probs)
            # use only new location for training
            # keep = pad_sequence([torch.tensor(e).to(device) for e in trg_nov], batch_first=True)
            keep = pad_sequence([torch.ones(e, dtype=torch.float32).to(device) for e in ds], batch_first=True)
            loss = torch.sum(loss * keep) / torch.sum(torch.tensor(ds).to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            processed_batch += 1
            batch_iterator.set_postfix_str(f"loss={loss.item():.4f}")

        epoch_time = Time.time() - start_time
        # print("epoch {:>2d} completed.".format(epoch_idx + 1))
        # print("time taken: {:.2f} sec".format(epoch_time))
        # print("avg. loss: {:.4f}".format(running_loss / processed_batch))

        print("epoch={:d}, loss={:.4f}".format(epoch_idx + 1, running_loss / processed_batch))
        model.save('./save/Bri@KNNsig--epoch'+str(epoch_idx)+'.pth')


        hr, ndcg, r5, r10, r15, r20 = evaluate(model, test_dataset, test_sampler, region_processer, loc2quadkey, device,
                                               config['test']['batch_size'],
                                               neg_given=bool(args.eval_samples))
        print("r@5: {:.4f},r@10: {:.4f},r@15: {:.4f},r@20: {:.4f}, ".format(r5, r10, r15, r20))
        print(" NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}".format(ndcg[4], ndcg[9],
                                                                                          ndcg[14], ndcg[19]))

        if args.results_path:
            with open(args.results_path, 'a') as f:
                print("r@5: {:.4f},r@10: {:.4f},r@15: {:.4f},r@20: {:.4f}, ".format(r5, r10, r15, r20), file=f)
                print(" NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}".format(ndcg[4], ndcg[9],
                                                                                                  ndcg[14], ndcg[19]),
                      file=f)

    print("training completed!")

    # print("=====evaluation=====")
    # match_rate = semantic_overlap(model, train_dataset.idx2loc, 1)
    # print("Semantic Overlap@1: {:.4f}".format(match_rate))
    # match_rate = semantic_overlap(model, train_dataset.idx2loc, 5)
    # print("Semantic Overlap@5: {:.4f}".format(match_rate))
    # match_rate = semantic_overlap(model, train_dataset.idx2loc, 10)
    # print("Semantic Overlap@10: {:.4f}".format(match_rate))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = unserialize('./data/RNSconfig.json')
    # parser.add_argument('--config', type=str, required=True)
    # parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--eval_samples', type=str, default=False)
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--load_path', type=str, default='')
    # parser.add_argument('--load_path', type=str, default='./save/Bri@DNS-pairloss-epoch55.pth')
    # parser.add_argument('--load_path', type=str, default='')
    # eval 시
    # np.random.seed(1000)
    # torch.manual_seed(1000)

    #실행방법: 에폭 완료되었으면 load-path에 경로 넣고 돌리면 됨

    parser.add_argument('--results_path', type=str, default='./results/test result_Bri@'+ 'knn-final-' + '.txt')
    args = parser.parse_args()

    # config = unserialize(args.config)


    # filename_raw = os.path.join(args.dataset, "totalCheckins.txt")
    # filename_clean = os.path.join(args.dataset, "QuadKeyLSBNDataset.data")
    # use PopularitySampler does not use follow file
    # user2mc_filename = os.path.join(args.dataset, "reg_trans_pmc_model.pkl")
    # loc_query_tree_path = os.path.join(args.dataset, "loc_query_tree.pkl")
    # knn_wrmf_sample_prob_path = os.path.join(args.dataset, "knn_wrmf_sample_prob.pkl")

    user2mc_filename = "reg_trans_pmc_model.pkl"
    loc_query_tree_path = "./data/loc_query_tree_Bri.pkl"
    knn_wrmf_sample_prob_path = "knn_wrmf_sample_prob.pkl"

    filename_raw = "./data/dataset_UbiComp2016_Bri_for_GeoSAN.txt"
    filename_clean = './data/UbiComp2016_QuadKeyLSBNDataset_Bri.data'

    if not os.path.isfile(filename_clean):
        dataset = QuadKeyLBSNDataset(filename_raw)
        serialize(dataset, filename_clean)
    else:
        dataset = unserialize(filename_clean)

    if config["train"]["negative_sampler"] in {"UniformNegativeSampler", "KNNsig", "KNNSampler",
                                               "KNNWRMFSampler"}:
        user_visited_locs = get_visited_locs(dataset)

    train_dataset, test_dataset = dataset.split()
    region_processer = dataset.QUADKEY
    # pth -> txt
    # temp_model = torch.load('./results/Bri/GeoSAN@100.pth')
    # save_embed(temp_model, train_dataset.idx2loc, 1)
    if args.eval_samples:
        eval_samples = np.load(args.eval_samples)
        test_dataset = NegInclLSBNDataset(test_dataset, eval_samples)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda:0' )#if torch.cuda.is_available() else 'cpu'

    print('embedding dim:' + str(config['model']['hidden_dim_encoder']))
    args.save_path = './results/Bri@KNNsig-' + '.pth' #여기
    # user_embedding_dim = location_embedding_dim = time_embedding_dim = region_embedding_dim = embedding_dim

    print(train_dataset.n_loc)
    model = QuadKeyLocPredictor(
        nuser=train_dataset.n_user,
        nloc=train_dataset.n_loc,
        ntime=train_dataset.n_time,
        nquadkey=len(region_processer.vocab.itos),
        user_dim=int(config['model']['user_embedding_dim']),
        loc_dim=int(config['model']['location_embedding_dim']),
        # time_dim=int(config['model']['time_embedding_dim']),
        # reg_dim=int(config['model']['region_embedding_dim']),
        time_dim=int(config['model']['time_embedding_dim']),
        reg_dim=int(config['model']['region_embedding_dim']),
        nhid=int(config['model']['hidden_dim_encoder']),
        nhead_enc=int(config['model']['num_heads_encoder']),
        nhead_dec=int(config['model']['num_heads_decoder']),
        nlayers=int(config['model']['num_layers_encoder']),
        dropout=float(config['model']['dropout']),
        **config['model']['extra_config']
    )
    model.to(device)
    loss_fn = loss.__getattribute__(config['train']['loss'])()
    # loss_fn = loss.__getattribute__(config['train']['loss'])()

    if config["train"]["negative_sampler"] == "UniformNegativeSampler":
        sampler = neg_sampler.UniformNegativeSampler(
            user_visited_locs=user_visited_locs,
            exclude_visited=True,
            nloc=train_dataset.n_loc
        )

        # sampler = neg_sampler.UniformNegativeSampler(train_dataset.n_loc)

    elif config["train"]["negative_sampler"] == "RegionUniformSampler":
        user2mc = unserialize(user2mc_filename)
        sampler = neg_sampler.RegionUniformSampler(
            user2mc=user2mc,
            n_region=train_dataset.n_region,
            region2loc=train_dataset.region2loc,
            exclude_visited=True
        )
    elif config["train"]["negative_sampler"] == "RadiusSampler":
        loc_query_sys = LocQuerySystem()
        loc_query_sys.load(loc_query_tree_path)
        sampler = neg_sampler.RadiusSampler(
            n_loc=train_dataset.n_loc,
            query_sys=loc_query_sys,
            **config["train"]["negative_sampler_config"]
        )
    elif config["train"]["negative_sampler"] == "KNNSampler":
        loc_query_sys = LocQuerySystem()
        loc_query_sys.load(loc_query_tree_path)
        sampler = neg_sampler.KNNSampler(
            query_sys=loc_query_sys,
            user_visited_locs=user_visited_locs,
            **config["train"]["negative_sampler_config"]
        )
    # elif config["train"]["negative_sampler"] == "KNN":
    #     loc_query_sys = LocQuerySystem()
    #     loc_query_sys.load(loc_query_tree_path)
    #     sampler = neg_sampler.KNNsig(
    #         query_sys=loc_query_sys,
    #         loc2freq=train_dataset.locidx2freq,
    #         user_visited_locs=user_visited_locs,
    #         **config["train"]["negative_sampler_config"]
    #     )
    elif config["train"]["negative_sampler"] == "KNNWRMFSampler":
        loc_query_sys = LocQuerySystem()
        loc_query_sys.load(loc_query_tree_path)
        knn_wrmf_prob = joblib.load(knn_wrmf_sample_prob_path)
        sampler = neg_sampler.KNNWRMFSampler(
            query_sys=loc_query_sys,
            user_knn_probs=knn_wrmf_prob['user_knn_probs'],
            user_knn_cum_probs=knn_wrmf_prob['user_knn_cum_probs'],
            user_visited_locs=user_visited_locs,
            **config["train"]["negative_sampler_config"]
        )
    elif config["train"]["negative_sampler"] == "PopularitySampler":
        sampler = neg_sampler.PopularitySampler(
            loc2freq=train_dataset.locidx2freq,
            user_visited_locs=user_visited_locs,
            exclude_visited=True
        )

    if config['optimizer']['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config['optimizer']['learning_rate']),
                                     betas=(0.9, 0.98))
    loc_query_sys = LocQuerySystem()
    loc_query_sys.load(loc_query_tree_path)

    test_sampler =  neg_sampler.testSampler(train_dataset.n_loc
        )

    # KNNtestSampler(
    #     query_sys=loc_query_sys,
    #     user_visited_locs=user_visited_locs,
    #     **config["train"]["negative_sampler_config"]
    # )

    if args.load_path:
        model.load(args.load_path)
        print("=====evaluation=====")
        hr, ndcg,r5, r10, r15, r20 = evaluate(model, test_dataset, test_sampler, region_processer, dataset.loc2quadkey, device,
                            config['test']['batch_size'],
                            neg_given=bool(args.eval_samples))
        print("{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f} ".format(r5, r10, r15, r20,ndcg[4], ndcg[9],ndcg[14], ndcg[19]))


        if args.results_path:
            with open(args.results_path, 'wt') as f:
                print("{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f} ".format(r5, r10, r15, r20, ndcg[4],
                                                                                        ndcg[9], ndcg[14], ndcg[19]),file=f)
        exit()

    train(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        optimizer=optimizer,
        loss_fn=loss_fn,
        negative_sampler=sampler,
        test_sampler=test_sampler,
        region_processer=region_processer,
        loc2quadkey=dataset.loc2quadkey,
        device=device,
        num_neg=config['train']['num_negative_samples'],
        batch_size=config['train']['batch_size'],
        num_epochs=config['train']['num_epochs'],
        writer=None,
        batch_size_test=config['test']['batch_size'],
        num_neg_test=config['test']['num_negative_samples'],
        test_neg_given=bool(args.eval_samples),
        num_workers=config['train']['num_workers'],
        save_results=args.results_path
    )

    if args.save_path:
        model.save(args.save_path)
