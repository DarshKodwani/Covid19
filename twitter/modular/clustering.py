# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 20:11:38 2020

@author: tharshi
"""
#%% Import Frameworks and Set-up

# import useful frameworks
import numpy as np
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt

import itertools
fr_it = itertools.chain.from_iterable

from sklearn.manifold import TSNE
import hdbscan

plt.style.use('seaborn')

print('Loading encoding...', end='')
X_en = np.load('features.npy')
print('Done.\n')

#%% Plot 2D projection

print('Reducing dimension...\n')
p = 5000
tsne = TSNE(perplexity=p, verbose=2, n_jobs=-1, n_iter=500)
X_t = tsne.fit_transform(X_en)
print('Done.\n')

#%%
min_size = 30
hddb = hdbscan.HDBSCAN(min_cluster_size=min_size)
labels = hddb.fit_predict(X_t)
K = len(np.unique(labels))

# define the colormap
cmap = plt.cm.rainbow
# extract all colors from the color map
cmaplist = [cmap(i) for i in range(cmap.N)]
# create the new map
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

# define the bins and no rmalize
bounds = np.linspace(0, K, K + 1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)    

fig3  = plt.figure()
ax3 = plt.subplot(111)
scat = ax3.scatter(X_t[:, 0],
                    X_t[:, 1],
                    c=labels,
                    cmap=cmap,
                    norm=norm,
                    s=1)
# create the colorbar
cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
cb.set_label('Classes')
cb.ax.set_alpha(1)
plt.draw()
plt.xlabel('Perplexity = {}, Minimum Cluster Size = {}'.format(p, min_size))
ax3.set_title('Tweets - tSNE')
ax3.set_yticklabels([])
ax3.set_xticklabels([])

print('Saving plot...', end='')
plt.savefig("embedding_tSNE_{}_{}.png".format(p, min_size))
print('Done.\n')