import torch
from torchvision import datasets
import networkx as nx
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np


dataset = datasets.MNIST('data', download=True)

img = np.array(dataset[100][0])/255.
ver_diff = img[:-1, :] - img[1:, :]
hor_diff = img[:, :-1] - img[:, 1:]

ver_cost = 1-ver_diff**2
hor_cost = 1-hor_diff**2
up_cost = 1-(img)**2
down_cost = 1-(1-img)**2

node_names = np.array([(f'{i:02d}_{j:02d}')
                       for i, j in np.ndindex(img.shape)]
                      ).reshape(img.shape)

g = nx.Graph()
for i, j in np.ndindex(img.shape):
    g.add_edge(node_names[i, j], 'top', capacity=up_cost[i, j])
    g.add_edge(node_names[i, j], 'bottom', capacity=down_cost[i, j])

for i, j in np.ndindex(img[:-1, :].shape):
    g.add_edge(node_names[i, j], node_names[i+1, j], capacity=ver_cost[i, j])

for i, j in np.ndindex(img[:, :-1].shape):
    g.add_edge(node_names[i, j], node_names[i, j+1], capacity=hor_cost[i, j])

value, partition = nx.minimum_cut(
    g,
    'top',
    'bottom')

out = np.zeros_like(img)
for n in [_ for _ in partition[1] if _ not in ['top', 'bottom']]:
    i = int(n[:2])
    j = int(n[-2:])
    out[i, j] = 1.

fig, axs = plt.subplots(3, 2)
for ax in axs.ravel():
    ax.axis('off')

axs[0, 0].imshow(img, cmap='gray')
axs[0, 1].imshow(out, cmap='gray')
axs[1, 0].imshow(up_cost, cmap='gray')
axs[1, 1].imshow(down_cost, cmap='gray')
axs[2, 0].imshow(ver_cost, cmap='gray')
axs[2, 1].imshow(hor_cost, cmap='gray')

plt.show(block=True)
