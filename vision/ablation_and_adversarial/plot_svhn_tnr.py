import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator

# data
n = [i for i in range(1,21)]
svhn_tnr  = np.load("CIFAR10_SVHN_tnr_diff_n_20.npz")['tnr'][:20]
# svhn_tnr = svhn_tnr*100.

# Create plots with pre-defined labels.
# Create plots with pre-defined labels.
fig, ax = plt.subplots()
ax.plot(n, svhn_tnr, linewidth=6, label='SVHN')

ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# plt.xlabel('n', fontsize=20) 
# plt.ylabel('TNR (90% TPR)', fontsize=20) 
plt.yticks(fontsize=25) 
plt.xticks(fontsize=25) 

#legend = ax.legend(fancybox=True, framealpha=0.5)
legend = ax.legend(frameon=False, fontsize=35)

# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('C0')

#plt.show()
plt.savefig('svhn_tnr.pdf')
