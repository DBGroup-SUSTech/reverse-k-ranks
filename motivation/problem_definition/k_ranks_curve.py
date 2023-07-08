import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as mticker
from matplotlib.ticker import ScalarFormatter


def show_curve(x, y, title_l):
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
    for i in range(len(x)):
        ax.plot(x[i], y[i], marker=marker_l[i], markersize=4, linestyle='solid',
                color=color_l[i],
                label=title_l[i])
    new_x = np.arange(1, 100, 10)
    new_y = np.zeros(10) + 152
    plt.plot(new_x, new_y, '--', color='#000000')

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

    ylabels = ax.get_yticks().tolist()
    print(ylabels)
    ylabels.append(153)
    ylabels = np.sort(ylabels)
    ax.yaxis.set_major_locator(mticker.FixedLocator(ylabels))  # 定位到散点图的x轴
    ax.set_yticklabels(['%d' % x for x in ylabels])  # 使用列表推导式循环将刻度转换成浮点数
    # plt.show()

    # 使用legend绘制多条曲线
    # ax.legend(loc='upper left', title="legend")
    ax.legend(loc='upper left')
    plt.title("k-ranks-curve")

    plt.xlabel("k")
    plt.ylabel("ranks")
    ax.grid(True, linestyle='-.')
    # plt.xticks([0, 0.1, 0.2, 0.3, 0.4])
    # plt.yticks([0.75, 0.8, 0.85])
    plt.show()
    plt.savefig('curve.jpg')
    plt.close()


if __name__ == '__main__':
    # line_idx_l = [16, 21]
    line_idx_l = [85, 1]
    rank_df = pd.read_csv('/home/bianzheng/Reverse-kRanks/result/movielens-small-DiskIndexBruteForce-rank.csv',
                          header=None)
    movie_df = pd.read_csv('/home/bianzheng/Dataset/MIPS/Reverse-kRanks/problem-definition/movielens-small/popular.csv')
    # query_item_idx_l = np.loadtxt(
    #     '/home/bianzheng/Dataset/MIPS/Reverse-kRanks/problem-definition/movielens-small/query_item.txt', dtype=np.int)

    # x_idx_l = np.array([2 ** i for i in range(9)], dtype=np.int)
    # x_idx_l = np.array([1, 2, 3, 4, 5, 6], dtype=np.int)
    x_idx_l = np.arange(1, 100, 10)
    y = []
    title_l = []
    for li in line_idx_l:
        ranks = np.array(rank_df.iloc[li - 1])
        title = movie_df.iloc[li - 1]['title']
        title_l.append(title)
        y.append(ranks[x_idx_l - 1])

    # random_line_idx_l = np.random.permutation(800)[:3] + 200
    # # random_line_idx_l = [234]
    # movie_df = pd.read_csv('/home/bianzheng/Dataset/MIPS/Reverse-kRanks/problem-definition/movielens-27m/random.csv')
    # query_item_idx_l = np.loadtxt(
    #     '/home/bianzheng/Dataset/MIPS/Reverse-kRanks/problem-definition/movielens-27m/query_item.txt', dtype=np.int)
    #
    # for li in random_line_idx_l:
    #     ranks = np.array(rank_df.iloc[li - 1])
    #     title = movie_df.iloc[li - 200 - 1]['title']
    #     title_l.append(title)
    #     y.append(ranks[x_idx_l - 1])
    # x = np.tile(x_idx_l, (len(line_idx_l) + len(random_line_idx_l), 1))

    x = np.tile(x_idx_l, (len(line_idx_l), 1))
    y = np.array(y, dtype=np.int)
    print(x)
    print(y)
    print(title_l)
    show_curve(x, y, title_l)
