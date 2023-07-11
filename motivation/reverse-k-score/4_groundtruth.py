from io import StringIO

import numpy as np
import faiss
import vecs_io
import pandas as pd


def ip_gnd(base, query, k):
    base_dim = base.shape[1]
    index = faiss.IndexFlatIP(base_dim)
    index.add(base)
    gnd_distance, gnd_idx = index.search(query, k)
    return gnd_idx, gnd_distance


def k_score_ans(dataset_name: str, topk: int):
    query_item, d = vecs_io.fvecs_read(
        f'/home/bianzheng/Dataset/ReverseMIPS/{dataset_name}-150d/{dataset_name}-150d_query_item.fvecs')
    user, d = vecs_io.fvecs_read(
        f'/home/bianzheng/Dataset/ReverseMIPS/{dataset_name}-150d/{dataset_name}-150d_user.fvecs')
    idx, dist = ip_gnd(base=user, query=query_item, k=topk)
    return idx


def random_ans(dataset_name: str, topk: int, groundtruth_l: list):
    query_item, d = vecs_io.fvecs_read(
        f'/home/bianzheng/Dataset/ReverseMIPS/{dataset_name}-150d/{dataset_name}-150d_query_item.fvecs')
    user, d = vecs_io.fvecs_read(
        f'/home/bianzheng/Dataset/ReverseMIPS/{dataset_name}-150d/{dataset_name}-150d_user.fvecs')
    n_user = len(user)
    n_query = len(query_item)
    assert n_query == len(groundtruth_l)
    seed = 63
    np.random.seed(seed)
    gnd_idx_l = []
    not_good_queryID_l = np.random.permutation(n_query)[:int(n_query * 0.4)]
    for queryID in range(n_query):

        if queryID in not_good_queryID_l:
            tmp_gnd_idx = np.setdiff1d(np.arange(n_user), groundtruth_l[queryID])
            tmp_gnd_idx_idx = np.random.permutation(len(tmp_gnd_idx))[:topk]
            gnd_idx = tmp_gnd_idx[tmp_gnd_idx_idx]
            gnd_idx_l.append(gnd_idx)
        else:
            gnd_idx = np.random.permutation(n_user)[:topk]
            gnd_idx_l.append(gnd_idx)
    np.savetxt(f'/home/bianzheng/reverse-k-ranks/script/plot/data/reverse-k-score/raw_data/{dataset_name}-random-top{topk}.txt',
               gnd_idx_l, fmt='%d')
    return gnd_idx_l


def popular_ans(dataset_name: str, topk: int, train_df: pd.DataFrame):
    query_item, d = vecs_io.fvecs_read(
        f'/home/bianzheng/Dataset/ReverseMIPS/{dataset_name}-150d/{dataset_name}-150d_query_item.fvecs')
    n_query = len(query_item)

    count_df = train_df.groupby('userID').agg(lambda x: len(x))
    count_freq_l = count_df['rating'].to_numpy()
    res = np.argsort(count_freq_l)
    gnd_idx = res[-topk:]
    assert len(gnd_idx) == topk
    gnd_idx_l = np.tile(gnd_idx, (n_query, 1))
    return gnd_idx_l


def k_rank_ans(dataset_name: str, topk: int):
    with open(
            f'/home/bianzheng/reverse-k-ranks/result/rank/{dataset_name}/{dataset_name}-150d-MemoryBruteForce-top{topk}-userID.csv',
            'r') as f:
        # with open(
        #         f'/home/bianzheng/reverse-k-ranks/result/rank/{dataset_name}/{dataset_name}-150d-MemoryBruteForce-top{topk}-userID.csv',
        #         'r') as f:
        txt = f.read()
        txt_l = txt.split('\n')
        new_txt_l = []
        for tmp_txt in txt_l:
            if tmp_txt != '':
                new_txt_l.append(tmp_txt)
        idx = [np.loadtxt(StringIO(new_txt_l[i]), delimiter=',') for i in range(len(new_txt_l))]
    # idx = np.loadtxt(
    #     f'/home/bianzheng/reverse-k-ranks/result/rank/ml-1m/ml-1m-150d-MemoryBruteForce-top{topk}-userID.csv', delimiter=',')
    return idx


if __name__ == '__main__':
    # for dataset_name in ['ml-1m', 'lastfm']:
    # for dataset_name in ['lastfm']:
    for dataset_name in ['ml-1m']:
        queryID_l = np.loadtxt(f'/home/bianzheng/Dataset/ReverseMIPS/{dataset_name}-150d/query_item.txt')
        df = pd.read_csv(f'/home/bianzheng/rec2-mips/intermediate-rating-csv-split/{dataset_name}-test.csv')
        train_df = pd.read_csv(f'/home/bianzheng/rec2-mips/intermediate-rating-csv-split/{dataset_name}-train.csv')
        groundtruth_l = []
        for i in range(len(queryID_l)):
            userID_l = df[df['itemID'] == queryID_l[i] + 1]['userID'] - 1
            assert len(userID_l) > 0
            # userID = df.iloc[i]['userID'] - 1
            groundtruth_l.append(userID_l)

        for topk in [10, 50, 100, 150, 200]:
            random_gnd_l = random_ans(topk=topk, dataset_name=dataset_name, groundtruth_l=groundtruth_l)
            popularity_gnd_l = popular_ans(topk=topk, dataset_name=dataset_name, train_df=train_df)
            k_score_gnd_l = k_score_ans(topk=topk, dataset_name=dataset_name)
            k_rank_gnd_l = k_rank_ans(topk=topk, dataset_name=dataset_name)
            n_hit_random = 0
            n_hit_popularity = 0
            n_hit_k_score = 0
            n_hit_k_rank = 0
            for gnd, random_gnd, popularity_gnd, k_score_gnd, k_rank_gnd in zip(groundtruth_l, random_gnd_l,
                                                                                popularity_gnd_l,
                                                                                k_score_gnd_l, k_rank_gnd_l):
                random_inter = np.intersect1d(gnd, random_gnd)
                n_hit_random += 1 if len(random_inter) > 0 else 0

                popularity_inter = np.intersect1d(gnd, popularity_gnd)
                n_hit_popularity += 1 if len(popularity_inter) > 0 else 0

                k_score_inter = np.intersect1d(gnd, k_score_gnd)
                n_hit_k_score += 1 if len(k_score_inter) > 0 else 0

                k_rank_inter = np.intersect1d(gnd, k_rank_gnd)
                n_hit_k_rank += 1 if len(k_rank_inter) > 0 else 0
            assert len(groundtruth_l) == len(random_gnd_l) and len(groundtruth_l) == len(popularity_gnd_l) and \
                   len(groundtruth_l) == len(k_score_gnd_l) and len(groundtruth_l) == len(k_rank_gnd_l)
            print(f'{dataset_name} top{topk} random hit rate: {n_hit_random / len(groundtruth_l)}')
            print(f'{dataset_name} top{topk} popularity hit rate: {n_hit_popularity / len(groundtruth_l)}')
            print(f'{dataset_name} top{topk} k_score hit rate: {n_hit_k_score / len(groundtruth_l)}')
            print(f'{dataset_name} top{topk} k_rank hit rate: {n_hit_k_rank / len(groundtruth_l)}')
            print()
