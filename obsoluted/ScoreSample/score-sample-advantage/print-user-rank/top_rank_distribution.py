import matplotlib.pyplot as plt
import numpy as np

rank_l = np.loadtxt('../../result/attribution/print-user-rank-top-70-movielens-27m.csv', delimiter=',')
scatter_x = rank_l[:, 0]

topk = len(rank_l[0])
scatter_y = rank_l[:, topk - 1] - rank_l[:, 0]

# make the data
# np.random.seed(3)
# x = 4 + np.random.normal(0, 2, 24)
# y = 4 + np.random.normal(0, 2, len(x))
# size and color:
# sizes = np.random.uniform(15, 80, len(x))
# colors = np.random.uniform(15, 80, len(x))

# movielens
n_user = 283228
n_data_item = 53889
#netflix
# n_user = 480189
# n_data_item = 17770
#yelp-small
# n_user = 500000
# n_data_item = 50000

# plot
fig, ax = plt.subplots()

for i in range(int(np.floor(n_data_item / 512))):
       sample_rank = i * 512
       plot_x = np.arange(sample_rank)
       plot_y = sample_rank - plot_x
       ax.plot(plot_x, plot_y, linestyle='dotted')

# ax.scatter(x, y, s=sizes, c=colors, vmin=0, vmax=100)
# ax.scatter(x, y, vmin=0, vmax=n_data_item)
ax.scatter(scatter_x, scatter_y, s=2)

# ax.set(xlim=(0, n_data_item),
#        ylim=(0, n_data_item))
ax.set(xlim=(0, 5000),
       ylim=(0, 5000))

ax.set_xlabel('top-1 rank')
ax.set_ylabel('top-{} difference'.format(topk))
ax.set_title('movielens-27m, n_user: {}, n_data_item {}'.format(n_user, n_data_item))

# plt.show()
plt.savefig('top-rank-distribution.jpg')
plt.close()
