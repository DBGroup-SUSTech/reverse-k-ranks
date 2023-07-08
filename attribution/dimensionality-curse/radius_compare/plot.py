import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from bisect import bisect_left
import matplotlib


def scatter():
    dataset_name = 'yelp'
    df = pd.read_csv(
        '../../../result/attribution/dimensionality_curse/DimensionalityCurse-amazon-home-kitchen-3d-Balltree-data_item-performance.txt')
    print(df)

    n_data_item = df.iloc[0, 0]
    print(n_data_item)
    np.linspace()

    scatter_x = df['num_descendant']
    scatter_y = df['size']
    print("corr", np.corrcoef(scatter_x, scatter_y))

    # plot
    fig, ax = plt.subplots()

    # ax.scatter(x, y, s=sizes, c=colors, vmin=0, vmax=100)
    # ax.scatter(x, y, vmin=0, vmax=n_data_item)
    ax.scatter(scatter_x, scatter_y, s=2)

    # ax.set(xlim=(0, n_data_item),
    #        ylim=(0, n_data_item))
    # ax.set(xlim=(0, 5000),
    #        ylim=(0, 5000))

    ax.set_xlabel('# items in a node')
    ax.set_ylabel('radius')

    # plt.show()
    plt.savefig('random-corrleation.jpg', bbox_inches='tight', dpi=600)
    plt.close()


linestyle_l = ['_', '-', '--', ':']
marker_l = ['x', "v", "o", "D", "s"]
markersize = 15

hatch_l = ['//', '\\', '||', '++']
# style_l = [(None, '#ffffff'), ('\\', '#ffffff'), ('//', '#ffffff'), (None, '#000000')]
style_l = [(None, '#ffffff'), ('\\', '#ffffff'), (None, '#000000')]

matplotlib.rcParams.update({'font.size': 15,
                            # 'text.usetex': True,
                            # "font.family": "Helvetica"
                            })


def process_data(sample_info: list, addr: str, method_name: str):
    df = pd.read_csv(addr, comment='#')
    scatter_x = df['num_descendant']
    scatter_y = df['size']
    print("corr", np.corrcoef(scatter_x, scatter_y))

    # print(df)
    bucket_value_l = sample_info[0]
    error_l = sample_info[1]

    bucket = np.zeros(len(bucket_value_l), dtype=np.float32)
    bucket_count = np.zeros(len(bucket_value_l), dtype=np.int32)
    for i in range(len(df)):
        num_descendant = df.iloc[i, 0]
        radius = df.iloc[i, 1]
        for bucket_value, error, j in zip(bucket_value_l, error_l, np.arange(len(bucket_value_l))):
            if bucket_value - error <= num_descendant <= bucket_value + error:
                bucket[j] += radius
                bucket_count[j] += 1
                break

    bucket = bucket / bucket_count

    if method_name == 'Rtree':
        bucket /= 2
    print(method_name, bucket)
    print(method_name, bucket_value_l)
    return bucket, bucket_value_l


def run(*, sample_info: list, dataset: str, method_name: str, plot_m: dict):
    bucket3, arr3 = process_data(sample_info=sample_info[0],
                                 addr='../../result/attribution/dimensionality_curse/'
                                      f'DimensionalityCurse-{dataset}-3d-{method_name}-data_item-performance.txt',
                                 method_name=method_name)
    # print(bisect_left(arr, 11))
    bucket30, arr30 = process_data(sample_info=sample_info[1],
                                   addr='../../result/attribution/dimensionality_curse/'
                                        f'DimensionalityCurse-{dataset}-30d-{method_name}-data_item-performance.txt',
                                   method_name=method_name)
    bucket300, arr300 = process_data(sample_info=sample_info[2],
                                     addr='../../result/attribution/dimensionality_curse/'
                                          f'DimensionalityCurse-{dataset}-300d-{method_name}-data_item-performance.txt',
                                     method_name=method_name)

    # plot
    fig, ax = plt.subplots()

    ax.plot(arr3, bucket3, color='#000000', linewidth=2.5, linestyle='-',
            label='3',
            marker=marker_l[0], fillstyle='none', markersize=markersize)
    ax.plot(arr30, bucket30, color='#000000', linewidth=2.5, linestyle='-',
            label='30',
            marker=marker_l[1], fillstyle='none', markersize=markersize)
    ax.plot(arr300, bucket300, color='#000000', linewidth=2.5, linestyle='-',
            label='300',
            marker=marker_l[2], fillstyle='none', markersize=markersize)

    ax.legend(frameon=True, loc='lower right')

    ax.set(xlim=plot_m['xlim'],
           ylim=plot_m['ylim'])

    ax.set_xlabel('# items in a node')
    ax.set_xscale('log')
    ax.set_ylabel('radius')

    # plt.show()
    plt.savefig(f'dimensionality-curse-{dataset}-{method_name}.jpg', bbox_inches='tight', dpi=600)
    plt.close()


if __name__ == '__main__':
    start_point = 30
    end_point = -1
    # first is the array that need to partition into, second is the error bound
    method_m = {
        # ['amazon-home-kitchen', 'Balltree']:
        #     [([], []),
        #      ([], []),
        #      ([], [])],
        'yelp-Balltree':
            [([75, 745, 3300, 50306, 159585], [5, 5, 300, 5, 5]),  # 3
             ([93, 840, 8500, 28671, 159585], [1, 10, 500, 5, 5]),  # 30
             ([88, 805, 4918, 43531, 159585], [0, 1, 1, 1, 5])],  # 300
        # ['amazon-home-kitchen', 'Rtree']:
        #     [(),
        #      ([28, 750, 17000, 42470, 409243], [6, 300, 3200, 10, 10]),
        #      ([30, 1050, 36000, 47847, 409243], [10, 100, 1000, 10, 10])],
        'yelp-Rtree':
            [([22, 544, 1199, 14915, 159585], [0, 1, 1, 1, 10]),  # 3
             ([22, 628, 1579, 15981, 159585], [1, 0, 1, 1, 10]),  # 30
             ([22, 1120, 36268, 51645, 159585], [1, 0, 1, 10, 10])],  # 300
    }

    # first is xlim, second is ylim
    plot_m = {
        'yelp-Balltree': {
            'xlim': (4.5e1, 3e5),
            'ylim': (0, 3.5),
        },
        'yelp-Rtree': {
            'xlim': (1.5e1, 3e5),
            'ylim': (1, 7),
        }
    }
    for dataset in ['yelp']:
        for method_name in ['Balltree', 'Rtree']:
            key = f'{dataset}-{method_name}'
            run(sample_info=method_m[key], dataset=dataset, method_name=method_name, plot_m=plot_m[key])
