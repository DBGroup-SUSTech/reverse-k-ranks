import numpy as np
import pandas as pd

method_name = 'topt_ip'

df_nodiff = pd.DataFrame([],
                         columns=['queryID', 'n_candidate', 'n_compute', 'prune_ratio'],
                         dtype=np.int32)
df_diff = pd.DataFrame([[-1, 0, 0, 0, 0]],
                       columns=['queryID', 'inner_product', 'memory_index_search',
                                'exact_rank_refinement_time', 'read_disk_time'], dtype=np.int32)

with open('test_amazon_{}.txt'.format(method_name), 'r') as f:
    lines = f.read().split("\n")
    n_query = 10
    for queryID in range(n_query):
        n_candidate = int(lines[queryID * 3].split(" ")[-3])
        n_compute = int(lines[queryID * 3 + 1].split(" ")[-3])
        read_disk_time = float(lines[queryID * 3 + 2].split(" ")[-1][:-1])
        exact_rank_refinement_time = float(lines[queryID * 3 + 2].split(" ")[-3][:-2])
        memory_index_search = float(lines[queryID * 3 + 2].split(" ")[-5][:-2])
        inner_product = float(lines[queryID * 3 + 2].split(" ")[-9][:-2])
        prune_ratio = float(lines[queryID * 3 + 2].split(" ")[-14][:-1])

        new_df_nodiff = pd.DataFrame([{'queryID': queryID,
                                       'n_candidate': n_candidate, 'n_compute': n_compute, 'prune_ratio': prune_ratio}])
        new_df_diff = pd.DataFrame([{'queryID': queryID, 'inner_product': inner_product,
                                     'memory_index_search': memory_index_search,
                                     'exact_rank_refinement_time': exact_rank_refinement_time,
                                     'read_disk_time': read_disk_time}])
        df_nodiff = pd.concat([df_nodiff, new_df_nodiff])
        df_diff = pd.concat([df_diff, new_df_diff])
        pass
df_diff.set_index('queryID', inplace=True)
df_diff = df_diff.diff()
df_diff.drop([-1], inplace=True)
df_nodiff.set_index('queryID', inplace=True)
df = df_diff.join(df_nodiff, on=['queryID'], how='inner')
df.reset_index(inplace=True)
order_l = ['queryID', 'n_candidate', 'n_compute', 'prune_ratio',
           'inner_product', 'memory_index_search',
           'exact_rank_refinement_time', 'read_disk_time']
df = df[order_l]
df.to_csv('{}.csv'.format(method_name), index=False, float_format="%.5f")
print(df)

mean_df = df.mean()
mean_df.to_csv('{}-mean.csv'.format(method_name), index=True, float_format="%.5f")
print(df.mean())

