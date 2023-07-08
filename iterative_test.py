import gen_fake_data
import run_rkr


def generate_data():
    gen_data_dir = '/home/bianzheng/Dataset/ReverseMIPS'

    n_query_item = 100
    n_dim = 30
    n_data_item = 5000
    n_user = 1000

    gen_fake_data.gen_data_uniform(n_user, n_data_item, n_query_item, n_dim, f"fake-uniform-{n_dim}d", gen_data_dir)
    gen_fake_data.gen_data_normal(n_user, n_data_item, n_query_item, n_dim, f"fake-normal-{n_dim}d", gen_data_dir)

    n_query_item = 100
    n_dim = 30
    n_data_item = 5000
    n_user = 5000

    gen_fake_data.gen_data_independent(n_user, n_data_item, n_query_item, n_dim, f'fakebig-{n_dim}d', gen_data_dir)


if __name__ == '__main__':
    while True:
        generate_data()
        dataset_l = [('fake-normal', 30), ('fake-uniform', 30), ('fakebig', 30)]
        topk_l = [10, 20, 30]
        index_dir = "/home/bianzheng/reverse-k-ranks/index"
        dataset_dir = "/home/bianzheng/Dataset/ReverseMIPS"

        run_rkr.run(dataset_l=dataset_l, topk_l=topk_l, index_dir=index_dir, dataset_dir=dataset_dir)
