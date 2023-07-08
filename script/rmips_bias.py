import faiss
import numpy as np
from script import vecs_io
import multiprocessing
import time
import os
import matplotlib.pyplot as plt


def parallel_reverse_mips_gnd(gnd_idx, query_idx_l):
    start_time = time.time()

    share_score_table = multiprocessing.Manager().list()
    for _ in range(len(query_idx_l)):
        share_score_table.append(0)

    manager = multiprocessing.managers.BaseManager()
    manager.register('RMIPSGnd', RMIPSGnd)
    manager.start()
    parallel_obj = manager.RMIPSGnd(gnd_idx, query_idx_l, multiprocessing.cpu_count())
    res_l = []
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    for i in range(multiprocessing.cpu_count()):
        res = pool.apply_async(r_mips_parallel, args=(parallel_obj, share_score_table, i))
        res_l.append(res)
    pool.close()
    pool.join()

    end_time = time.time()
    print("get reverse MIPS time: %.2f" % (end_time - start_time))
    return share_score_table


class RMIPSGnd:
    def __init__(self, gnd_idx, query_idx_l, total_process):
        self.gnd_idx = gnd_idx
        self.query_idx_l = query_idx_l
        self.total_process = total_process

    def get_share_data(self):
        return self.gnd_idx, self.query_idx_l, self.total_process


def r_mips_parallel(obj, share_score_table, start_idx):
    gnd_idx, query_idx_l, total_process = obj.get_share_data()
    # iteration for every query
    query_len = len(query_idx_l)
    for i in range(start_idx, query_len, total_process):
        if i % 50 == 0:
            print("get reverse mips result " + str(i))
        # tmp_score_table = np.zeros(shape=n_item, dtype=np.float32)
        r_mips_gnd = []
        idx = query_idx_l[i]
        for j, mips_gnd in enumerate(gnd_idx, 0):
            if idx in mips_gnd:
                r_mips_gnd.append(j)
        share_score_table[i] = r_mips_gnd
    print("finish parallel")


def ip_gnd(base, query, k):
    base_dim = base.shape[1]
    index = faiss.IndexFlatIP(base_dim)
    index.add(base)
    gnd_distance, gnd_idx = index.search(query, k)
    return gnd_idx, gnd_distance


def show_curve(x, y, fig_name):
    # 第一个是横坐标的值，第二个是纵坐标的值
    # plt.figure(num=3, figsize=(8, 5))
    # marker
    # o 圆圈, v 倒三角, ^ 正三角, < 左三角, > 右三角, 8 大圆圈, s 正方形, p 圆圈, * 星号, h 菱形, H 六面体, D 大菱形, d 瘦的菱形, P 加号, X 乘号
    # 紫色#b9529f 蓝色#3953a4 红色#ed2024 #231f20 深绿色#098140 浅绿色#7f8133 #0084ff
    # solid dotted

    marker_l = ['H', 'D', 'P', '>', '*', 'X', 's', '<', '^', 'p', 'v']
    color_l = ['#b9529f', '#3953a4', '#ed2024', '#098140', '#231f20', '#7f8133', '#0084ff']
    plt.plot(x, y, marker=marker_l[0], linestyle='solid',
             color=color_l[0],
             label='label')

    # 使用legend绘制多条曲线
    # plt.title('graph kmeans vs knn')
    plt.legend(loc='upper left')
    plt.title("Proportion of the items with non-empty result set")

    plt.xlabel("parameter k")
    plt.ylabel("proportion of users recommended(%)")
    plt.grid(True, linestyle='-.')
    # plt.xticks([0, 0.1, 0.2, 0.3, 0.4])
    # plt.yticks([0.75, 0.8, 0.85])
    plt.savefig('%s.jpg' % fig_name)
    plt.close()


if __name__ == '__main__':
    topk_l = [10, 30, 50, 70, 90, 110]

    # ds_l = ['movielens-small', 'movielens-20m', 'movielens-25m', 'movielens-27m', 'netflix', 'yahoomusic']
    ds_l = ['netflix']
    for dataset in ds_l:
        base_dir = '/home/bianzheng/Dataset/Reverse-MIPS'
        item_dir = os.path.join(base_dir, dataset, '%s_item.fvecs' % dataset)
        user_dir = os.path.join(base_dir, dataset, '%s_user.fvecs' % dataset)
        item_l, d = vecs_io.fvecs_read(item_dir)
        user_l, d = vecs_io.fvecs_read(user_dir)
        print("item shape", item_l.shape, "user shape", user_l.shape)
        query_idx_dir = os.path.join(base_dir, dataset, 'query_idx.txt')
        query_idx_l = np.loadtxt(query_idx_dir)

        gnd_len_percent_l = []
        for topk in topk_l:
            total_gnd_idx, total_gnd_dis = ip_gnd(item_l, user_l, topk)

            rmips_result = parallel_reverse_mips_gnd(total_gnd_idx, query_idx_l)

            rmips_res_len_l = [len(_) for _ in rmips_result]

            cnt_array = np.where(rmips_res_len_l, 0, 1)
            nonempty_cnt = np.sum(cnt_array)

            gnd_len_percent_l.append((len(rmips_res_len_l) - nonempty_cnt) / len(rmips_res_len_l) * 100)
        show_curve(topk_l, gnd_len_percent_l, dataset)
