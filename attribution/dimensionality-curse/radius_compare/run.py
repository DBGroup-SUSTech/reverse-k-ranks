import numpy as np
import pandas as pd

if __name__ == '__main__':
    dataset = 'yelp'
    method_name = 'Rtree'
    addr = '/home/bianzheng/reverse-k-ranks/script/plot/data/dimensionality_curse/DimensionalityCurse-yelp-300d-Rtree-data_item-performance.txt'
    df = pd.read_csv(addr)
    n_item = np.sum(df[df['height'] == 13]['n_element'])
    print(n_item)

    pass
