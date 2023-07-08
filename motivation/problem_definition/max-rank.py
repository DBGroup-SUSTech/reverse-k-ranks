import numpy as np
import pandas as pd

if __name__ == '__main__':
    rank_df = pd.read_csv('/home/bianzheng/Reverse-kRanks/result/movielens-small-DiskIndexBruteForce-rank.csv',
                          header=None)
    movie_df = pd.read_csv('/home/bianzheng/Dataset/MIPS/Reverse-kRanks/problem-definition/movielens-small/popular.csv')
    # query_item_idx_l = np.loadtxt(
    #     '/home/bianzheng/Dataset/MIPS/Reverse-kRanks/problem-definition/movielens-small/query_item.txt', dtype=np.int)
    trun_idx = 100
    rank_df = rank_df[:100]
    idx = np.argmax(rank_df[trun_idx])
    print(idx)
    print(movie_df.iloc[idx])

    idx = np.argmin(rank_df[trun_idx])
    print(idx)
    print(movie_df.iloc[idx])
