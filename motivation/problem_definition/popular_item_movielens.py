import numpy as np
import os
import pandas as pd

# 找到最出名的item编号

if __name__ == '__main__':
    dataset_l = ['movielens-small']
    base_dir = '/home/bianzheng/Dataset/Recommendation/raw-data/'
    for ds in dataset_l:
        ratings_fname = os.path.join(base_dir, ds, 'ratings.csv')
        rating_l = pd.read_csv(ratings_fname)
        # 找到5分中最多的几个item
        rating_l = rating_l[rating_l['rating'] >= 5]
        rating_l = rating_l.groupby('movieId').count()
        rating_l.reset_index(inplace=True, drop=False)
        del rating_l['userId'], rating_l['timestamp']
        rating_l.rename(columns={'rating': 'rating_count'}, inplace=True)
        rating_l.sort_values('rating_count', ascending=False, inplace=True)
        # print(rating_l[:10])
        rating_l = rating_l[:20]
        
        title_fname = os.path.join(base_dir, ds, 'movies.csv')
        title_l = pd.read_csv(title_fname)
        merge_l = pd.merge(rating_l, title_l, how='inner', left_on=['movieId'], right_on=['movieId'])
        print(merge_l)
