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


def k_rank_ans(dataset_name: str, topk: int):
    with open(
            f'/home/bianzheng/reverse-k-ranks/result/rank/{dataset_name}-150d-MemoryBruteForce-top{topk}-userID.csv',
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
    for dataset_name in ['ml-1m', 'lastfm']:
    # for dataset_name in ['lastfm']:
        queryID_l = np.loadtxt(f'/home/bianzheng/Dataset/ReverseMIPS/{dataset_name}-150d/query_item.txt')
        df = pd.read_csv(f'/home/bianzheng/rec2-mips/intermediate-rating-csv-split/{dataset_name}-test.csv')
        groundtruth_l = []
        for i in range(len(queryID_l)):
            userID = df.iloc[i]['userID'] - 1
            groundtruth_l.append(userID)

        for topk in [10, 50, 100, 150, 200]:
            k_score_gnd_l = k_score_ans(topk=topk, dataset_name=dataset_name)
            k_rank_gnd_l = k_rank_ans(topk=topk, dataset_name=dataset_name)
            n_hit_k_score = 0
            n_hit_k_rank = 0
            for gnd, k_score_gnd, k_rank_gnd in zip(groundtruth_l, k_score_gnd_l, k_rank_gnd_l):
                k_score_inter = np.intersect1d(gnd, k_score_gnd)
                n_hit_k_score += 1 if len(k_score_inter) > 0 else 0

                k_rank_inter = np.intersect1d(gnd, k_rank_gnd)
                n_hit_k_rank += 1 if len(k_rank_inter) > 0 else 0
            assert len(groundtruth_l) == len(k_score_gnd_l) and len(groundtruth_l) == len(k_rank_gnd_l)
            print(f'{dataset_name} top{topk} k_score hit rate: {n_hit_k_score / len(groundtruth_l)}')
            print(f'{dataset_name} top{topk} k_rank hit rate: {n_hit_k_rank / len(groundtruth_l)}')
