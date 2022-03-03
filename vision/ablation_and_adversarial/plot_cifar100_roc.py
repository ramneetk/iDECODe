import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
#mpl.rcParams['text.usetex'] = True

from matplotlib.ticker import MaxNLocator


# data
n = [i for i in range(1,21)]
cifar100_roc = np.load("CIFAR10_CIFAR100_roc_diff_n_20.npz")['roc'][:20]

# Create plots with pre-defined labels.
# Create plots with pre-defined labels.
fig, ax = plt.subplots()
plt.plot(n, cifar100_roc, linewidth=6, label='CIFAR100')

ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))


#plt.xlabel(r'$|\mathcal V(x)|$', fontsize=20) 
#plt.ylabel('AUROC', fontsize=20) 
plt.yticks(fontsize=25) 
plt.xticks(fontsize=25) 

#legend = ax.legend(fancybox=True, framealpha=0.5)
legend = ax.legend(frameon=False, fontsize=35)

# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('C0')

#plt.show()
plt.savefig('cifar100_roc.pdf')

