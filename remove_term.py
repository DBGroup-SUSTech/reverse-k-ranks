import os


def rename_performance_file(performance_file: str):
    # loop through all files in the directory
    for file in os.listdir(performance_file):
        # check if the file name contains the term "retrieval"
        if "retrieval" in file:
            # construct the new file name by removing the first occurrence of the term "retrieval"
            new_file_name = file.replace("retrieval-", "", 1)
            for dataset_name in ['amazon-home-kitchen', 'movielens-1b', 'yahoomusic_big', 'yelp']:
                if dataset_name in new_file_name:
                    # construct the new file name by removing the first occurrence of the term "retrieval"
                    new_file_name2 = new_file_name.replace(dataset_name, f"{dataset_name}-150d", 1)
                    # rename the file
                    os.rename(os.path.join(performance_file, file), os.path.join(performance_file, new_file_name2))
                    break


def rename_index_file(index_file: str):
    # loop through all files in the directory
    for file in os.listdir(index_file):
        # check if the file name contains the term "retrieval"
        for dataset_name in ['amazon-home-kitchen', 'movielens-1b', 'yahoomusic_big', 'yelp']:
            if dataset_name in file:
                # construct the new file name by removing the first occurrence of the term "retrieval"
                new_file_name2 = file.replace(dataset_name, f"{dataset_name}-150d", 1)
                # rename the file
                os.rename(os.path.join(index_file, file), os.path.join(index_file, new_file_name2))
                break


def rename_raw_data_name(data_dir: str, dataset_name: str, dimension: int):
    new_data_filename = os.path.join(data_dir, f'{dataset_name}-{dimension}d')
    os.rename(os.path.join(data_dir, dataset_name), new_data_filename)
    os.rename(os.path.join(new_data_filename, f'{dataset_name}_data_item.fvecs'),
              os.path.join(new_data_filename, f'{dataset_name}-{dimension}d_data_item.fvecs'))

    os.rename(os.path.join(new_data_filename, f'{dataset_name}_query_item.fvecs'),
              os.path.join(new_data_filename, f'{dataset_name}-{dimension}d_query_item.fvecs'))

    os.rename(os.path.join(new_data_filename, f'{dataset_name}_user.fvecs'),
              os.path.join(new_data_filename, f'{dataset_name}-{dimension}d_user.fvecs'))


if __name__ == "__main__":
    project_dir = '/home/zhengbian/reverse-k-ranks'
    performance_file = os.path.join(project_dir, 'result', 'vis_performance')
    single_query_performance_file = os.path.join(project_dir, 'result', 'single_query_performance')
    rank_file = os.path.join(project_dir, 'result', 'rank')

    memory_index_file = os.path.join(project_dir, 'index/memory_index')
    qrs_to_sample_index_file = os.path.join(project_dir, 'index/qrs_to_sample_index')
    query_distribution_file = os.path.join(project_dir, 'index/query_distribution')
    data_dir = '/home/zhengbian/Dataset/ReverseMIPS'

    rename_performance_file(performance_file=performance_file)
    rename_index_file(index_file=single_query_performance_file)
    rename_index_file(index_file=rank_file)

    for dataset_prefix in ['amazon-home-kitchen', 'movielens-1b', 'yahoomusic_big', 'yelp']:
        os.rename(os.path.join(project_dir, 'index', f'{dataset_prefix}.index'),
                  os.path.join(project_dir, 'index', f'{dataset_prefix}-150d.index'))

    rename_index_file(index_file=memory_index_file)
    rename_index_file(index_file=qrs_to_sample_index_file)
    rename_index_file(index_file=query_distribution_file)
    for dataset_tuple in [('amazon-electronics', 150), ('amazon-home-kitchen', 150), ('amazon-office-products', 150),
                          ('fake-normal', 30), ('fake-uniform', 30), ('fakebig', 30), ('movielens-1b', 150),
                          ('yahoomusic_big', 150), ('yelp', 150)]:
        dataset_name, dimension = dataset_tuple
        rename_raw_data_name(data_dir=data_dir, dataset_name=dataset_name, dimension=dimension)
