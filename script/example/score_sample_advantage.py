import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
import matplotlib

params = {
    'axes.labelsize': 8,
    'font.size': 8,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.usetex': False,
    'figure.figsize': [4.5, 4.5]
}
linestyle_l = ['_', '-', '--', ':']
color_l = ['#3D0DFF', '#6BFF00', '#00E8E2', '#EB0225', '#FF9E03']
marker_l = ['H', 'D', 'P', '>', '*', 'X', 's', '<', '^', 'p', 'v']
markersize = 10
matplotlib.RcParams.update(params)

# 29, 25, 23, 20, 19, 17, 15, 12, 11, 10, 7
n = 10000
arr = np.random.normal(loc=0, scale=1, size=n)
sigma = 5
mu = 18
scale_arr = sigma * arr + mu
score_sample_l = np.array([30, 22, 14, 6])
rank_sample_l = np.array([29, 20, 15])

plot_min = 1
plot_max = 5000

appr_ss_l = [(scale_arr > score).sum() for score in score_sample_l]
appr_rs_l = [(scale_arr > score).sum() for score in rank_sample_l]

fig, ax = plt.subplots()


def prune_ratio(x, appr_l, n):
    # assert plot_min <= x <= plot_max
    idx = np.argmax(appr_l >= x)
    appr = appr_l[idx]
    return 1 - appr / n


for i, appr in enumerate(appr_ss_l, 0):
    if appr > plot_max:
        x_l = np.geomspace(appr_ss_l[i - 1] + 1, plot_max, num=2, endpoint=True)
        y_l = [prune_ratio(num, appr_ss_l, n) for num in x_l]
        ax.plot(x_l, y_l, color='#343434', linestyle='-', marker=marker_l[0], markersize=markersize,
                label="Score Sample")
        break
    if i == 0:
        x_l = np.geomspace(0.1, appr, num=5, endpoint=True)
        y_l = [prune_ratio(num, appr_ss_l, n) for num in x_l]
        print(x_l, y_l)
        ax.plot(x_l, y_l, color='#343434', linestyle='-', marker=marker_l[0], markersize=markersize)
    else:
        x_l = np.geomspace(appr_ss_l[i - 1] + 1, appr_ss_l[i], num=5, endpoint=True)
        y_l = [prune_ratio(num, appr_ss_l, n) for num in x_l]
        ax.plot(x_l, y_l, color='#343434', linestyle='-', marker=marker_l[0], markersize=markersize)

for i, appr in enumerate(appr_rs_l, 0):
    if appr > plot_max:
        x_l = np.geomspace(appr_rs_l[i - 1] + 1, plot_max, num=2, endpoint=True)
        y_l = [prune_ratio(num, appr_rs_l, n) for num in x_l]
        ax.plot(x_l, y_l, color='#343434', linestyle='-', marker=marker_l[1], markersize=markersize,
                label="Rank Sample")
        break
    if i == 0:
        x_l = np.geomspace(0.1, appr, num=5, endpoint=True)
        y_l = [prune_ratio(num, appr_rs_l, n) for num in x_l]
        ax.plot(x_l, y_l, color='#343434', linestyle='-', marker=marker_l[1], markersize=markersize)
    else:
        x_l = np.geomspace(appr_rs_l[i - 1] + 1, appr_rs_l[i], num=5, endpoint=True)
        y_l = [prune_ratio(num, appr_rs_l, n) for num in x_l]
        ax.plot(x_l, y_l, color='#343434', linestyle='-', marker=marker_l[1], markersize=markersize)

asym_x_l = np.arange(plot_min, plot_max)
asym_y_l = [1 - k / n for k in asym_x_l]
ax.plot(asym_x_l, asym_y_l, color='#343434', linestyle='dotted', label="Theoretical Minimum")
plt.xlim(plot_min, plot_max)
plt.ylim(0.5, 1)

ax.legend(frameon=False, loc='lower right')

plt.xscale("log")
# plt.yscale("log")

ax.xaxis.set_major_formatter(ScalarFormatter())


def PercentageFormat(y, pos):
    # Find the number of decimal places required
    decimalplaces = int(np.maximum(-np.log10(y), 0))  # =0 for numbers >=1
    # Insert that number into a format string
    # formatstring = '{}%'.format(decimalplaces * 100)
    formatstring = '{{:.{:1d}f}}%'.format(decimalplaces)
    # Return the formatted tick label
    return formatstring.format(y * 100)


def DecimalPlacesFormat(y, pos):
    # Find the number of decimal places required
    decimalplaces = int(np.maximum(-np.log10(y), 0))  # =0 for numbers >=1
    # Insert that number into a format string
    formatstring = '{{:.{:1d}f}}'.format(decimalplaces)
    # Return the formatted tick label
    return formatstring.format(y)


# ax.yaxis.set_major_formatter(ticker.FuncFormatter(DecimalPlacesFormat))

plt.xlabel(r'$k$')
plt.ylabel('Refinement Ratio')

plt.savefig('ScoreSampleAdvantage.pdf', bbox_inches='tight')
plt.close()
print("appr_ss_l", appr_ss_l)
print("appr_rs_l", appr_rs_l)
