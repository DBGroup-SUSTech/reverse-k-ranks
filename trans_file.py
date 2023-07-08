import os


def trans_file(index_dir, dataset_name, n_sample):
    os.system("scp -r -P 22 "
              f"zhengbian@10.16.5.247:/home/zhengbian/reverse-k-ranks/{index_dir}/memory_index/QueryRankSampleSearchKthRank-{dataset_name}-n_sample_{n_sample}-n_sample_query_5000-sample_topk_600.index  "
              "/home/zhengbian/index_reverse_k_ranks/memory_index/ ")
    os.system("scp -r -P 22 "
              f"zhengbian@10.16.5.247:/home/zhengbian/reverse-k-ranks/{index_dir}/memory_index/RankSample-{dataset_name}-n_sample_{n_sample}.index  "
              "/home/zhengbian/index_reverse_k_ranks/memory_index/ ")
    print(f"trans file success {index_dir} {dataset_name} {n_sample}")


if __name__ == '__main__':
    movielens_l = [947, 1895, 3791, 7582, 15164, 52127]
    for n_sample in movielens_l:
        trans_file('index', 'movielens-27m', n_sample)

    yahoomusic_l = [147, 294, 588, 1177, 2355, 4711, 8097]
    for n_sample in yahoomusic_l:
        trans_file('index', 'yahoomusic_big', n_sample)

    yelp_l = [122, 245, 490, 980, 1961, 3923, 6743]
    for n_sample in yelp_l:
        trans_file('index', 'yelp', n_sample)

    # amazon_l = [106, 213, 427, 855, 1710]
    # for n_sample in movielens_l:
    #     trans_file('index_amazon', 'amazon-home-kitchen', n_sample)

    username = 'zhengbian'
    dataset_dir = f'/home/{username}/Dataset/ReverseMIPS'
    index_dir = f'/home/{username}/reverse-k-ranks/index'
    dataset_name = 'amazon-home-kitchen'
    os.system(f'cd build && ./bst --dataset_dir {dataset_dir} --dataset_name {dataset_name} --index_dir {index_dir}')
