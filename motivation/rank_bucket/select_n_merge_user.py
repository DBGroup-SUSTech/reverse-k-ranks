# 使用聚类的loss确定是否使用很多个

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import vecs_io
import matplotlib.ticker as mticker


def show_curve(x, y):
    # 第一个是横坐标的值，第二个是纵坐标的值
    # plt.figure(num=3, figsize=(8, 5))
    # marker
    # o 圆圈, v 倒三角, ^ 正三角, < 左三角, > 右三角, 8 大圆圈, s 正方形, p 圆圈, * 星号, h 菱形, H 六面体, D 大菱形, d 瘦的菱形, P 加号, X 乘号
    # 紫色#b9529f 蓝色#3953a4 红色#ed2024 #231f20 深绿色#098140 浅绿色#7f8133 #0084ff
    # solid dotted

    marker_l = ['H', 'D', 'P', '>', '*', 'X', 's', '<', '^', 'p', 'v']
    color_l = ['#b9529f', '#3953a4', '#ed2024', '#098140', '#231f20', '#7f8133', '#0084ff']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, marker=marker_l[0], markersize=4, linestyle='solid',
            color=color_l[0],
            # label=title_l[0])
            )

    # ax.set_xticks(x)
    # ax.set_xscale('log', base=2)
    # ax.xaxis.set_major_formatter(ScalarFormatter())

    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    # ax.set_xlim(0, 30)
    # ax.set_xticklabels('%d' % i for i in range(1, 21, 2))

    xlabels = ax.get_xticks().tolist()
    print(xlabels)
    ax.xaxis.set_major_locator(mticker.FixedLocator(xlabels))  # 定位到散点图的x轴
    ax.set_xticklabels(['%d' % x for x in xlabels])  # 使用列表推导式循环将刻度转换成浮点数

    # ylabels = ax.get_yticks().tolist()
    # print(ylabels)
    # ylabels = np.sort(ylabels)
    # ax.yaxis.set_major_locator(mticker.FixedLocator(ylabels))  # 定位到散点图的x轴
    # ax.set_yticklabels(['%d' % x for x in ylabels])  # 使用列表推导式循环将刻度转换成浮点数
    # plt.show()

    # 使用legend绘制多条曲线
    # ax.legend(loc='upper left', title="legend")
    # ax.legend(loc='upper left')
    plt.title("choose of n_merge_user")

    plt.xlabel("n_merge_user")
    plt.ylabel("loss")
    ax.grid(True, linestyle='-.')
    # plt.xticks([0, 0.1, 0.2, 0.3, 0.4])
    # plt.yticks([0.75, 0.8, 0.85])
    plt.show()
    plt.savefig('curve.jpg')
    plt.close()


if __name__ == '__main__':
    data, dim = vecs_io.dvecs_read('/home/bianzheng/Dataset/MIPS/Reverse-kRanks/movielens-27m/movielens-27m_user.dvecs')
    cls_l = [2 ** i for i in range(1, 12)]
    rmse_l = []
    for n_cls in cls_l:
        kmeans = KMeans(n_clusters=n_cls, random_state=0).fit(data)
        labels = kmeans.labels_
        cls_center = kmeans.cluster_centers_
        rmse = np.average(np.array([np.linalg.norm(cls_center[labels[i]] - data[i]) for i in range(len(data))]))
        rmse_l.append(rmse)
        print("%d complete" % n_cls)
    print(rmse_l)
    show_curve(cls_l, rmse_l)
