import vecs_io
import numpy as np
import matplotlib.pyplot as plt

# 对数据进行svd的处理


if __name__ == '__main__':
    # dataset_l = ['netflix', 'yahoomusic']
    dataset_l = ['netflix', 'movielens-20m']
    for ds in dataset_l:
        base, dim = vecs_io.fvecs_read('/home/bianzheng/attribution/data/%s/%s_item.fvecs' % (ds, ds))
        query, dim = vecs_io.fvecs_read('/home/bianzheng/attribution/data/%s/%s_user.fvecs' % (ds, ds))
        base = base.T

        u, sigma, v_prime = np.linalg.svd(base, full_matrices=False)
        v = v_prime.T
        # print(u.shape, sigma.shape, v.shape)
        sigma = np.diag(sigma)
        # print(np.dot(np.dot(u, sigma), v.T))
        # print(base)

        # 保持原有的query不变
        b1 = base
        q1 = query.T
        # 变化query, 就是乘上sigma和u
        q2 = np.array([np.dot(np.dot(sigma, u.T), q) for q in query]).T
        b2 = v.T
        # 变换query, 只是乘上u
        q3 = np.array([np.dot(u.T, q) for q in query]).T
        b3 = np.dot(sigma, v.T)

        # print(b1.shape, q1.shape)
        # print(b2.shape, q2.shape)
        # print(b3.shape, q3.shape)
        b2 = b2.T
        q2 = q2.T
        print(ds, b2.shape, q2.shape)
        vecs_io.fvecs_write('/home/bianzheng/attribution/data/%s/%s_item_fexipro_svd.fvecs' % (ds, ds), b2)
        vecs_io.fvecs_write('/home/bianzheng/attribution/data/%s/%s_user_fexipro_svd.fvecs' % (ds, ds), q2)

        b3 = b3.T
        q3 = q3.T
        print(ds, b3.shape, q3.shape)
        vecs_io.fvecs_write('/home/bianzheng/attribution/data/%s/%s_item_svd.fvecs' % (ds, ds), b3)
        vecs_io.fvecs_write('/home/bianzheng/attribution/data/%s/%s_user_svd.fvecs' % (ds, ds), q3)
