import vecs_io
import os
import numpy as np

'''first is n_item, second is n_query, third is n_user'''
dataset_m = {'fake-normal': [5000, 100, 1000],
             'fake-uniform': [5000, 100, 1000],
             'fakebig': [5000, 100, 5000],
             'fakehuge': [100000, 100, 100000],
             'fakehuge2': [100000, 100, 200000],
             'fakehuge3': [200000, 100, 100000],

             'ml-1m': [3706, 100, 6040],
             'lastfm': [17632, 100, 1892],

             'movielens-1b': [272038, 1000, 2197225],
             'yahoomusic_big': [135736, 1000, 1823179],
             'yelp': [159585, 1000, 2189457],
             'amazon-electronics': [475002, 1000, 4201696],
             'amazon-home-kitchen': [409243, 1000, 2511610],
             'amazon-office-products': [305800, 1000, 3404914],

             'amazon-home-kitchen_weighted_query': [409243, 1000, 2511610],
             'yelp_weighted_query': [159585, 1000, 2189457], }


def delete_file_if_exist(dire):
    if os.path.exists(dire):
        command = 'rm -rf %s' % dire
        print(command)
        os.system(command)


if __name__ == '__main__':
    dimensionality = 150
    username = 'zhengbian'
    dataset_prefix_l = ['amazon-home-kitchen', 'yelp']
    # dataset_prefix_l = ['yelp']
    # basic_dir = '/home/bianzheng/Dataset/ReverseMIPS'
    mips_dir = f'/home/{username}/Dataset/MIPS'
    rmips_dir = f'/home/{username}/Dataset/ReverseMIPS'
    n_query = 1000

    for dataset_prefix in dataset_prefix_l:
        dataset_name = f'{dataset_prefix}-{dimensionality}d-update-item'
        input_dir = os.path.join(mips_dir, f'{dataset_prefix}-{dimensionality}d')
        item, d = vecs_io.fvecs_read(
            os.path.join(input_dir, '%s_item.fvecs' % dataset_prefix))
        user, d = vecs_io.fvecs_read(
            os.path.join(input_dir, '%s_user.fvecs' % dataset_prefix))

        random_itemID_l = np.random.permutation(len(item))
        queryID_l = random_itemID_l[:n_query]
        n_remain_item = len(item) - n_query
        n_sample_item = int(n_remain_item / 1.02)
        data_itemID_l = random_itemID_l[n_query:n_query + n_sample_item]
        update_itemID_l = random_itemID_l[n_query + n_sample_item:]

        new_user = user
        new_data_item = item[data_itemID_l, :]
        new_query_item = item[queryID_l, :]
        new_update_item = item[update_itemID_l, :]

        output_dir = f'/home/{username}/Dataset/ReverseMIPS/{dataset_name}'
        delete_file_if_exist(output_dir)
        os.mkdir(output_dir)

        np.savetxt(os.path.join(output_dir, 'query_item.txt'), queryID_l, fmt="%d")
        np.savetxt(os.path.join(output_dir, 'data_item.txt'), data_itemID_l, fmt="%d")
        np.savetxt(os.path.join(output_dir, 'update_item.txt'), update_itemID_l, fmt="%d")

        vecs_io.fvecs_write(os.path.join(output_dir, '%s_data_item.fvecs' % dataset_name), new_data_item)
        vecs_io.fvecs_write(os.path.join(output_dir, '%s_data_item_update.fvecs' % dataset_name), new_update_item)
        vecs_io.fvecs_write(os.path.join(output_dir, '%s_query_item.fvecs' % dataset_name), new_query_item)
        vecs_io.fvecs_write(os.path.join(output_dir, '%s_user.fvecs' % dataset_name), new_user)

        print(f"{dataset_name}, n_user {len(new_user)}, n_data_item {len(new_data_item)}, n_query {len(new_query_item)}, n_update_item {len(new_update_item)}")

    for dataset_prefix in dataset_prefix_l:
        dataset_name = f'{dataset_prefix}-{dimensionality}d-update-user'
        input_dir = os.path.join(mips_dir, f'{dataset_prefix}-{dimensionality}d')
        item, d = vecs_io.fvecs_read(
            os.path.join(input_dir, '%s_item.fvecs' % dataset_prefix))
        user, d = vecs_io.fvecs_read(
            os.path.join(input_dir, '%s_user.fvecs' % dataset_prefix))

        random_itemID_l = np.random.permutation(len(item))
        queryID_l = random_itemID_l[:n_query]
        data_itemID_l = random_itemID_l[n_query:]

        random_userID_l = np.random.permutation(len(user))
        n_remain_user = int(len(user) / 1.02)
        userID_l = random_userID_l[:n_remain_user]
        update_userID_l = random_userID_l[n_remain_user:]

        new_user = user[userID_l, :]
        new_update_user = user[update_userID_l, :]
        new_data_item = item[data_itemID_l, :]
        new_query_item = item[queryID_l, :]

        output_dir = f'/home/{username}/Dataset/ReverseMIPS/{dataset_name}'
        delete_file_if_exist(output_dir)
        os.mkdir(output_dir)

        np.savetxt(os.path.join(output_dir, 'query_item.txt'), queryID_l, fmt="%d")
        np.savetxt(os.path.join(output_dir, 'data_item.txt'), data_itemID_l, fmt="%d")
        np.savetxt(os.path.join(output_dir, 'user.txt'), userID_l, fmt="%d")
        np.savetxt(os.path.join(output_dir, 'update_user.txt'), update_userID_l, fmt="%d")

        vecs_io.fvecs_write(os.path.join(output_dir, '%s_data_item.fvecs' % dataset_name), new_data_item)
        vecs_io.fvecs_write(os.path.join(output_dir, '%s_query_item.fvecs' % dataset_name), new_query_item)
        vecs_io.fvecs_write(os.path.join(output_dir, '%s_user.fvecs' % dataset_name), new_user)
        vecs_io.fvecs_write(os.path.join(output_dir, '%s_user_update.fvecs' % dataset_name), new_update_user)

        print(f"{dataset_name}, n_user {len(new_user)}, n_data_item {len(new_data_item)}, n_query {len(new_query_item)}, n_update_user {len(new_update_user)}")

