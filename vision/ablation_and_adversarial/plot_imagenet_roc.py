import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator

# data
n = [i for i in range(1,21)]
imagenet_roc = np.load("CIFAR10_IMAGENET_roc_diff_n_20.npz")['roc'][:20]

# Create plots with pre-defined labels.
# Create plots with pre-defined labels.
fig, ax = plt.subplots()
ax.plot(n, imagenet_roc, linewidth=6, label='IMAGENET')

ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))


# plt.xlabel('n', fontsize=20) 
# plt.ylabel('AUROC', fontsize=20) 
plt.yticks(fontsize=25) 
plt.xticks(fontsize=25) 

#legend = ax.legend(fancybox=True, framealpha=0.5)
legend = ax.legend(frameon=False, fontsize=35)

# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('C0')

# plt.show()
plt.savefig('imagenet_roc.pdf')
