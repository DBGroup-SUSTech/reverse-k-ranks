from script import vecs_io
import numpy as np
import matplotlib.pyplot as plt


# 统计在svd之后, 经过d维度占据了总共norm的百分之多少
# 作用在于, 经过了svd的处理之后, 我可以仅仅计算多少个维度就能覆盖大部分的bound
def dimension_norm(data):
    abs_data = np.abs(data)
    arr_sum = np.sum(abs_data, axis=1)
    # print(arr_sum)
    arr_cumsum = np.cumsum(abs_data, axis=1)
    # print(arr_cumsum)
    arr_sum = arr_sum[:, np.newaxis]
    arr_percent = arr_cumsum / arr_sum
    # print(arr_percent)
    arr_avg = np.average(arr_percent, axis=0)
    # print(arr_avg)
    return arr_avg


# 输入是需要道道
def show_hist(figname, dataset, base, query):
    arr_base = dimension_norm(base)
    arr_query = dimension_norm(query)

    plt.plot(np.linspace(0, 1, len(base[0])), arr_base, marker='H', linestyle='solid', color='#b9529f',
             label='base')
    plt.plot(np.linspace(0, 1, len(base[0])), arr_query, marker='P', linestyle='solid', color='#ed2024',
             label='query')
    plt.legend(loc='lower right', title=figname)

    plt.xlabel("the number of dimension")
    plt.ylabel("percentage")
    plt.grid(True, linestyle='-.')
    plt.title(dataset)
    plt.savefig("%s-%s.jpg" % (dataset, figname))
    plt.close()


if __name__ == '__main__':
    # dataset_l = ['netflix', 'yahoomusic']
    dataset_l = ['netflix']
    for ds in dataset_l:
        base, dim = vecs_io.fvecs_read('/home/bianzheng/Dataset/NIPS/%s/%s_base.fvecs' % (ds, ds))
        query, dim = vecs_io.fvecs_read('/home/bianzheng/Dataset/NIPS/%s/%s_query.fvecs' % (ds, ds))
        base = base.T

        u, sigma, v_prime = np.linalg.svd(base, full_matrices=False)
        # print(sigma)
        v = v_prime.T
        # print(u.shape, sigma.shape, v.shape)
        sigma = np.diag(sigma)
        # print(np.dot(np.dot(u, sigma), v.T))
        # print(base)

        # 保持原有的query不变
        q1 = query.T
        # 变化query, 就是乘上sigma和u
        q2 = np.array([np.dot(np.dot(sigma, u.T), q) for q in query]).T
        b2 = v.T
        # 变换query, 只是乘上u
        q3 = np.array([np.dot(u.T, q) for q in query]).T
        b3 = np.dot(sigma, v.T)

        # print(np.dot(q1.T, base))
        # print(np.dot(q2.T, b2))
        # print(np.dot(q3.T, b3))
        # print(q3.shape, b3.T.shape)
        show_hist('sigma-times-query', ds, b2.T, q2.T)
        show_hist('sigma-times-base', ds, b3.T, q3.T)
