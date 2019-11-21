import math
import scipy.sparse as sp
import numpy as np
import torch

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from sklearn.preprocessing import normalize
from sklearn.svm import SVC, LinearSVC, NuSVC
import  networkx as nx
import copy
import itertools
from collections import Counter



### For layers m###

def glorot(tensor):
    """initialization nn weights"""
    stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

def kaiming_uniform(tensor, fan, a):
    if tensor is not None:
        bound = math.sqrt(6 / ((1 + a**2) * fan))
        tensor.data.uniform_(-bound, bound)

def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


### Others ###

def getSimilarity(result):
    """getting dot similarity

    Parameter
    ---------
    result : scipy.sparse.spmatrix
             Sparse unweighted adjacency matrix

    Return : ,array-like, shape [result.shape[0],result.shape[0]]
            numpy matrix
    ------
    """
    similarity = result.dot(result.T)
    similarity.setdiag(values=0)
    #np.fill_diagonal(similarity,0)
    #print(similarity)
    similarity.eliminate_zeros()
    assert similarity.diagonal().sum() == 0
    return similarity.toarray()

def make_symmetric(sparse_matrix):
    """symmetric a sparse matrix"""
    sparse_matrix = sparse_matrix.tocsr()
    rows, cols = sparse_matrix.nonzero()
    sparse_matrix[cols, rows] = sparse_matrix[rows, cols]
    return sparse_matrix

def to_torch_sparse_tensor(M):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    M = M.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((M.row, M.col)).astype(np.float32)).long()
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    T = torch.sparse.FloatTensor(indices, values, shape)
    return T



# Two ways for normalization

def normalize(matrix):
    """Row-normalize sparse matrix"""
    rowsum = np.array(matrix.sum(1))
    r_inv = np.power(rowsum, -1.0).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(matrix)
    return mx

def symnormalize(adj):
    """Symmetrically normalize adjacency matrix."""
    # 图的归一化拉普拉斯矩阵
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()



def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

def evaluate(model, edges, adj, labels, mask):
    model.eval()
    with torch.no_grad():
        _, _, logits = model(edges, adj)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)

## metrics

def Wasserstein(mu, sigma, idx1, idx2):
    """
    metric between two Gaussian embeddings
    return: shape: (len(idx1),)
    """
    p1 = torch.sum(torch.pow((mu[idx1] - mu[idx2]), 2), 1)
    p2 = torch.sum(torch.pow(torch.pow(sigma[idx1], 1/2) - torch.pow(sigma[idx2], 1/2), 2), 1)
    return p1 + p2

def KLDivergence(mu, sigma, idx1=None, idx2=None):
    """
    metric between two Gaussian embeddings
    return: shape: (len(idx1),)
    """
    L = mu.shape[1]
    sigma_ratio = sigma[idx2] / sigma[idx1]
    trace_fac = torch.sum(sigma_ratio, 1)
    log_det = torch.sum(torch.log(sigma_ratio + 1e-14), 1)
    mu_diff_sq = torch.sum(torch.pow(mu[idx1] - mu[idx2], 2) / sigma[idx1], 1)
    return 0.5 * (trace_fac + mu_diff_sq - L - log_det)



def eliminate_self_loops(A):
    """Remove self-loops from the adjacency matrix."""
    A = A.tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A

def set_adj(adj, idx):
    diff = np.setdiff1d(range(adj.shape[0]),idx)
    te = adj.tolil()
    te[diff] = np.zeros([len(diff), adj.shape[1]])
    assert te[diff].sum() == 0
    return te

## Dropout for sparse matrix

def dropout_sparse_tensors(X, rate, training):
    if isinstance(X, torch.sparse.FloatTensor):
        values = torch.nn.functional.dropout(X._values(), p=rate, training=training)
        return torch.sparse.FloatTensor(X._indices(), values, X.shape)
    else:
        return torch.nn.functional.dropout(X, p=rate, training=training)

def softmax_sparse_tensor(X):
    if isinstance(X, torch.sparse.FloatTensor):
        values = torch.nn.functional.softmax(X._values())
        return torch.sparse.FloatTensor(X._indices(), values, X.shape)
    else:
        return torch.nn.functional.softmax(X)

## Referenced from  "Pitfalls of graph neural network evaluation"_benchmark

def largest_connected_components(sparse_graph, n_components=1):
    """Select the largest connected components in the graph.

    Parameters
    ----------
    sparse_graph : SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.

    Returns
    -------
    sparse_graph : SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.

    """
    _, component_indices = sp.csgraph.connected_components(sparse_graph.adj_matrix)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep
    ]
    return create_subgraph(sparse_graph, nodes_to_keep=nodes_to_keep)


def create_subgraph(sparse_graph, _sentinel=None, nodes_to_remove=None, nodes_to_keep=None):
    """Create a graph with the specified subset of nodes.

    Exactly one of (nodes_to_remove, nodes_to_keep) should be provided, while the other stays None.
    Note that to avoid confusion, it is required to pass node indices as named arguments to this function.

    Parameters
    ----------
    sparse_graph : SparseGraph
        Input graph.
    _sentinel : None
        Internal, to prevent passing positional arguments. Do not use.
    nodes_to_remove : array-like of int
        Indices of nodes that have to removed.
    nodes_to_keep : array-like of int
        Indices of nodes that have to be kept.

    Returns
    -------
    sparse_graph : SparseGraph
        Graph with specified nodes removed.

    """
    # Check that arguments are passed correctly
    if _sentinel is not None:
        raise ValueError("Only call `create_subgraph` with named arguments',"
                         " (nodes_to_remove=...) or (nodes_to_keep=...)")
    if nodes_to_remove is None and nodes_to_keep is None:
        raise ValueError("Either nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None and nodes_to_keep is not None:
        raise ValueError("Only one of nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None:
        nodes_to_keep = [i for i in range(sparse_graph.num_nodes()) if i not in nodes_to_remove]
    elif nodes_to_keep is not None:
        nodes_to_keep = sorted(nodes_to_keep)
    else:
        raise RuntimeError("This should never happen.")

    sparse_graph.adj_matrix = sparse_graph.adj_matrix[nodes_to_keep][:, nodes_to_keep]
    if sparse_graph.attr_matrix is not None:
        sparse_graph.attr_matrix = sparse_graph.attr_matrix[nodes_to_keep]
    if sparse_graph.labels is not None:
        sparse_graph.labels = sparse_graph.labels[nodes_to_keep]
    if sparse_graph.node_names is not None:
        sparse_graph.node_names = sparse_graph.node_names[nodes_to_keep]
    return sparse_graph

def remove_underrepresented_classes(g, train_examples_per_class, val_examples_per_class):
    """Remove nodes from graph that correspond to a class of which there are less than
    num_classes * train_examples_per_class + num_classes * val_examples_per_class nodes.

    Those classes would otherwise break the training procedure.
    """
    min_examples_per_class = train_examples_per_class + val_examples_per_class
    examples_counter = Counter(g.labels)
    keep_classes = set(class_ for class_, count in examples_counter.items() if count > min_examples_per_class)
    keep_indices = [i for i in range(len(g.labels)) if g.labels[i] in keep_classes]

    return create_subgraph(g, nodes_to_keep=keep_indices)

def add_self_loops(A, value=1.0):
    """Set the diagonal."""
    A = A.tolil()  # make sure we work on a copy of the original matrix
    A.setdiag(value)
    A = A.tocsr()
    if value == 0:
        A.eliminate_zeros()
    return A


def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes