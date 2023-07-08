import numpy as np
import matplotlib.pyplot as plt

data = []
item = np.random.normal(size=150)
for i in range(100000):
    u1 = np.random.normal(size=150)
    u2 = np.random.normal(size=150)
    u1 = u1 / np.linalg.norm(u1)
    u2 = u2 / np.linalg.norm(u2)
    x = np.dot(u1.T, item)
    y = np.dot(u2.T, item)
    data.append([x, y])
data = np.array(data)

scatter_x = data[:, 0]
scatter_y = data[:, 1]
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

ax.set_xlabel('x')
ax.set_ylabel('y')

# from matplotlib import ticker

# formatter = ticker.ScalarFormatter(useMathText=True)
# formatter.set_scientific(True)
# formatter.set_powerlimits((-1, 1))
# ax.yaxis.set_major_formatter(formatter)

# plt.ticklabel_format(axis='y', style='scientific', useMathText=True)

# ax.set_title('{}'.format(dataset))

# plt.show()
plt.savefig('random-corrleation.jpg')
plt.close()
