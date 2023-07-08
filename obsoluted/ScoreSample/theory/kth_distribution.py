#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as stats
import decimal

decimal.getcontext().prec = 100

n_user = 6000
n_item = 3000
# n_user = 283228
# n_item = 53889
k = 10

n = n_user
m = n_item
arr = []
std_norm_ins = stats.norm(0, 1)
for i in range(0, n + 2, 1):
    x = i
    combination = math.comb(n, k) * (n - k)
    gx = (2 * x - m) / math.sqrt(m)
    pdf = decimal.Decimal(std_norm_ins.pdf(gx))
    gx_cdf = decimal.Decimal(std_norm_ins.cdf(gx))
    cdf = pow(gx_cdf, n - k - 1) * pow((1 - gx_cdf), k)
    y = combination * pdf * cdf
    print("combination {:.3f}, gx {:.3f}, pdf {:.3f}, gx_cdf {:.3f}, cdf {:.3f}, x {}, y {:.3f}".format(
        combination, gx, pdf, gx_cdf, cdf, x, y))
    arr.append([x, y])
arr = np.array(arr)

print(arr)
np.savetxt('kth-distribution.txt', arr)

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
ax.plot(arr[:, 0], arr[:, 1], linestyle='solid', color='#000000', label="distribution")
plt.savefig("kth-distribution.jpg")
