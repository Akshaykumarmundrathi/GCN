# Graph Convolutional Networks — From Scratch on Cora

A from-scratch PyTorch implementation of a Graph Convolutional Network (Kipf & Welling,
2017), trained on the **Cora citation network**, with two follow-on experiments extending
the base architecture.

## Notebooks

| Notebook | What it does |
|---|---|
| [`GCN.ipynb`](GCN.ipynb) | Baseline: a single-hidden-layer GCN with a custom `GraphConvolution` layer (symmetric-normalized adjacency, one-hot label encoding), trained for 200 epochs on Cora's train/val/test split. |
| [`GCN_Task1_1.ipynb`](GCN_Task1_1.ipynb) | Extends the baseline to **two hidden layers**, adding a second `GraphConvolution` layer and updating the forward pass. |
| [`GCN_Task3_1.ipynb`](GCN_Task3_1.ipynb) | Builds on the two-layer model and adds `limit_neighbors()` — a neighbor-sampling step that caps each node's adjacency to a fixed number of neighbors, to study its effect on the graph convolution. |

## Implementation

Each notebook implements the core GCN machinery manually rather than relying on a graph
library:

- One-hot label encoding and row-normalization for node features
- Symmetric adjacency normalization (`D^-1/2 A D^-1/2`) via SciPy sparse matrices
- Sparse-to-dense conversion into PyTorch tensors
- A custom `GraphConvolution` layer (`nn.Module` subclass) with its own learnable weight matrix
- Standard train/eval loop reporting loss and accuracy per epoch

## Running

Each notebook is Colab-ready (open via the badge at the top) and expects the Cora dataset
as `GCN_export.zip` in the working directory, unzipped to `data/cora/`.

## Context

Coursework exploring how depth (hidden layers) and neighborhood size affect a graph
convolutional network's node-classification accuracy on a citation graph.
