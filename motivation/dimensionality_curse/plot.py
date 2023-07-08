import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import json


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
    ax.plot(x[0], y[0], marker=marker_l[0], linestyle='solid',
             color=color_l[0],
             label='hr@50')

    ax.plot(x[1], y[1], marker=marker_l[1], linestyle='solid',
             color=color_l[1],
             label='hr@100')

    ax.plot(x[2], y[2], marker=marker_l[2], linestyle='solid',
             color=color_l[2],
             label='hr@200')

    # ax.set_xticks(x)
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ScalarFormatter())

    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    # ax.spines['left'].set_position(('data', 0))
    # ax.spines['bottom'].set_position(('data', 0))

    # ax.annotate(r'd=50', xycoords='data', xy=(x[5], y[5]), textcoords="offset points", xytext=(10, -30),
    #             fontsize=14, arrowprops=dict(arrowstyle='-|>', connectionstyle='angle3', color='red'))
    # ax.annotate(r'd=100', xycoords='data', xy=(x[7], y[7]), textcoords="offset points", xytext=(10, -30),
    #             fontsize=14, arrowprops=dict(arrowstyle='-|>', connectionstyle='angle3', color='red'))
    # ax.annotate(r'd=150', xycoords='data', xy=(x[9], y[9]), textcoords="offset points", xytext=(10, -30),
    #             fontsize=14, arrowprops=dict(arrowstyle='-|>', connectionstyle='angle3', color='red'))

    # ax.text(x[5], y[5], '50')
    # ax.text(x[7], y[7], '100')
    # ax.text(x[9], y[9], '150')
    # ax.set_xlim(1, 500000)

    # 使用legend绘制多条曲线
    # ax.legend(loc='upper left', title="legend")
    ax.legend(loc='upper left')
    plt.title("dimensionality vs hitting rate")

    plt.xlabel("dimension")
    plt.ylabel("hitting rate")
    ax.grid(True, linestyle='-.')
    # plt.xticks([0, 0.1, 0.2, 0.3, 0.4])
    # plt.yticks([0.75, 0.8, 0.85])
    plt.show()
    # plt.savefig('curve.jpg')
    plt.close()


if __name__ == "__main__":
    x = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    hr_50_l = []
    hr_100_l = []
    hr_200_l = []
    for d in x:
        with open('result/hitting_rate-%d-new.json' % d, 'r') as f:
            tmp_json = json.load(f)
            tmp_hr = tmp_json['test_result']['hit@50']
            hr_50_l.append(tmp_hr)
            tmp_hr = tmp_json['test_result']['hit@100']
            hr_100_l.append(tmp_hr)
            tmp_hr = tmp_json['test_result']['hit@200']
            hr_200_l.append(tmp_hr)
    x_arr = np.tile(x, [3, 1])
    print(x_arr)
    y_arr = np.array([hr_50_l, hr_100_l, hr_200_l])
    show_curve(x_arr, y_arr)