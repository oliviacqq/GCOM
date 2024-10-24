import time
import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data, DataLoader
import os
import create_more
import pickle
import math
import sklearn.metrics
from sklearn.metrics import cluster

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0')

from torch_geometric.datasets import Planetoid

dataset_cora = Planetoid(root='./cora/', name='Cora')
# 打印数据集
print(dataset_cora)

# 提取data，并转换为device格式
data_cora = dataset_cora[0]

indices = np.arange(0, data_cora.y.shape[0])
label = data_cora.y[(data_cora.train_mask == 1) | (data_cora.test_mask == 1) | (data_cora.val_mask == 1)]
label_indices = indices[(data_cora.train_mask == 1) | (data_cora.test_mask == 1) | (data_cora.val_mask == 1)]



def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def make_all_dists(bin_adj, dmax, use_weights=False):
    g = nx.from_numpy_array(bin_adj.detach().numpy())
    if not use_weights:
        lengths = nx.shortest_path_length(g)
    else:
        lengths = nx.shortest_path_length(g, weight='weight')
    dist = torch.zeros_like(bin_adj)
    for u, lens_u in lengths:
        for v in range(bin_adj.shape[0]):
            if v in lens_u:
                dist[u,v] = lens_u[v]
            else:
                dist[u,v] = dmax
    return dist

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_nodes):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

        self.be = nn.BatchNorm1d(out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.bd = nn.BatchNorm1d(out_channels)

        self.fc_0 = nn.Linear(1,out_channels)
        self.fc_1 = nn.Embedding(2000, out_channels)
        self.fc_2 = nn.Linear(3*out_channels, out_channels)

    def forward(self, x, edge_index, edge_weight, edges, degree):
        x = self.conv1(x, edge_index,edge_weight)
        x = F.relu(x)
        x = F.dropout(x, 0)
        x = self.conv2(x, edge_index,edge_weight)
        x = self.bn(x)
        x = F.relu(x)

        edges = self.fc_0(edges)
        edges = self.be(edges)
        edges = F.relu(edges)

        degree = self.fc_1(degree)
        degree = self.bd(degree)
        degree = F.relu(degree)

        x_concat = torch.cat((x,edges,degree), dim=1)
        x_concat = self.fc_2(x_concat)
        return x_concat


class MocoModel(torch.nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, n_nodes, m=0.99, K=256):
        super().__init__()
        self.m = m
        self.K = K

        # Query network
        self.q_net = GCN(dim_in, dim_hidden, dim_out, n_nodes)

        # Key network
        self.k_net = GCN(dim_in, dim_hidden, dim_out, n_nodes)

        # Register a momentum version of the key encoder
        # self.k_net_momentum = GCN(dim_in, dim_hidden, dim_out)
        for param_q, param_k in zip(self.q_net.parameters(), self.k_net.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False    # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim_out, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_w", torch.randn(n_nodes, K))
        self.queue_w = nn.functional.normalize(self.queue_w, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, idx, x, edge_index, edge_weight, edge, degree, batch):
        # Compute query embeddings
        embs_q = self.q_net(x, edge_index, edge_weight, edge, degree)
        embs_q = F.normalize(embs_q, dim=1)
        # q = F.normalize(q, dim=1)
        if (batch == x.shape[0]):
            return embs_q
        q = embs_q[idx*batch:(idx+1)*batch,:] # n * c

        # Compute key embeddings
        with torch.no_grad():
            self._momentum_update_key_encoder()
            embs_k = self.k_net(x, edge_index, edge_weight, edge, degree)
            embs_k = F.normalize(embs_k, dim=1)
            k = embs_k[idx * batch:(idx + 1) * batch, :]  # n * c

        # Positive pairs
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        # Negative pairs
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= 0.07

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return embs_q, logits, labels

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.q_net.parameters(), self.k_net.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

def make_modularity_matrix(adj,v):
    adj = adj*(torch.ones(adj.shape[0], adj.shape[0]) - torch.eye(adj.shape[0]))
    degrees = adj.sum(dim=0).unsqueeze(1)
    mod = adj - v*degrees@degrees.t()/adj.sum()
    return mod

def calculate_internal_edges(adjacency_matrix, node_labels):
    num_nodes = adjacency_matrix.size(0)

    # 构造类别矩阵
    label_matrix = (node_labels.unsqueeze(0) == node_labels.unsqueeze(1)).to(device)

    # 构造边矩阵
    edge_matrix = torch.mul(adjacency_matrix, label_matrix).to(device)

    # 计算类别内部边数
    internal_edges = torch.sum(edge_matrix).item() // 2  # 除以2是因为邻接矩阵是对称的

    return internal_edges

def conductance(adjacency, clusters):
    """Computes graph conductance as in Yang & Leskovec (2012).
    Args:
        adjacency: Input graph in terms of its sparse adjacency matrix.
        clusters: An (n,) int cluster vector.
    Returns:
        The average conductance value of the graph clusters.
    """
    inter = 0  # Number of inter-cluster edges.
    intra = 0  # Number of intra-cluster edges.
    cluster_indices = torch.zeros(adjacency.shape[0], dtype=torch.bool)
    for cluster_id in torch.unique(clusters):
        cluster_indices[:] = False
        cluster_indices[torch.where(clusters == cluster_id)[0]] = True
        adj_submatrix = adjacency[cluster_indices, :]
        inter += torch.sum(adj_submatrix[:, cluster_indices])
        intra += torch.sum(adj_submatrix[:, ~cluster_indices])
    return intra / (inter + intra)

import torch.optim as optim

# adj_all, features, labels = load_data("./data/cora/","cora")
features = data_cora.x
labels = data_cora.y.to(device)
adj_all = to_dense_adj(data_cora.edge_index).squeeze()
# adj_all = data_cora.edge_index
n = adj_all.shape[0]
n_nodes = adj_all.shape[0]
bin_adj = adj_all.float()
dist_all = make_all_dists(bin_adj, 100)
diameter = dist_all[dist_all < 100].max()
# dist_all[dist_all == 100] = diameter
edge_index_all, edge_weight_all = torch_geometric.utils.dense_to_sparse(bin_adj)
# edge_weight_all = edge_weight_all.reshape((edge_weight_all.shape[0],-1))
dist_row = torch.sum(dist_all, dim=1)
dist_row = dist_row.unsqueeze(1)
diameter = dist_all.max()
degree = torch.sum(bin_adj, dim=0).to(device)
degree = degree.long()
mod = make_modularity_matrix(bin_adj,1).to(device)
mod_1 = make_modularity_matrix(bin_adj,1).to(device)
graph = torch_geometric.data.Data(x=features.to(device), edge_index=edge_index_all.to(device), edge_attr=edge_weight_all.to(device), dist_row=dist_row.to(device), degree=degree.to(device))
bin_adj = bin_adj.to(device)

internal_edges = calculate_internal_edges(bin_adj, labels)
# 确定矩阵的形状，行数等于输入tensor的长度，列数等于输入tensor的最大值加1
matrix_shape = (len(labels), labels.max() + 1)
# 使用torch.eye()创建单位矩阵，然后使用索引操作将每行替换为对应的值
output_tensor = torch.eye(matrix_shape[1])[labels]
output_tensor = output_tensor.to(device)
print("按照标签划分")
obj = (1. / bin_adj.sum()) * (output_tensor.t() @ mod_1 @ output_tensor).trace()
cudu = conductance(bin_adj,labels)
print("Q:",obj," C:",cudu)

model = MocoModel(graph.x.shape[1], 128, 64, n_nodes).to(device)
model.load_state_dict(torch.load('pre_cora_new.pth'))
K = 7

class MLP(torch.nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_in, dim_hidden), nn.ReLU(), nn.Linear(dim_hidden, dim_out)
        )

    def forward(self, embs):
        mlp_embs = self.fc(embs)
        return mlp_embs

class MLP1(torch.nn.Module):
    def __init__(self, model, dim_in, dim_hidden, dim_out):
        super().__init__()
        self.pretrained_model = model
        for param in self.pretrained_model.parameters():
            param.requires_grad = True
        self.fc = nn.Sequential(
            nn.Linear(dim_in, dim_hidden), nn.ReLU(), nn.Linear(dim_hidden, dim_out)
        )

    def forward(self, idx, x, edge_index, edge_weight, edge, degree, batch):
        embs = self.pretrained_model(0, x, edge_index, edge_weight, edge, degree, x.shape[0])
        mlp_embs = self.fc(embs)
        return mlp_embs

def rounding(x):
    '''
    Fast pipage rounding implementation for uniform matroid
    '''
    i = 0
    j = 1
    x = x.clone()
    for t in range(len(x)-1):
        if x[i] == 0 and x[j] == 0:
            i = max((i,j)) + 1
        elif x[i] + x[j] < 1:
            if np.random.rand() < x[i]/(x[i] + x[j]):
                x[i] = x[i] + x[j]
                x[j] = 0
                j = max((i,j)) + 1
            else:
                x[j] = x[i] + x[j]
                x[i] = 0
                i = max((i,j)) + 1
        else:
            if np.random.rand() < (1 - x[j])/(2 - x[i] - x[j]):
                x[j] = x[i] + x[j] - 1
                x[i] = 1
                i = max((i,j)) + 1

            else:
                x[i] = x[i] + x[j] - 1
                x[j] = 1
                j = max((i,j)) + 1
    return x

start = time.time()
model_mlp = MLP(64, 32, K).to(device)
model_mlp1 = MLP1(model, 64, 62, K).to(device)
optimizer = optim.Adam(model_mlp1.parameters(), lr=0.001, weight_decay=5e-4)


loss_fin = 0


def _pairwise_confusion(
    y_true,
    y_pred):
  """Computes pairwise confusion matrix of two clusterings.
  Args:
    y_true: An (n,) int ground-truth cluster vector.
    y_pred: An (n,) int predicted cluster vector.
  Returns:
    True positive, false positive, true negative, and false negative values.
  """
  contingency = cluster.contingency_matrix(y_true, y_pred)
  same_class_true = np.max(contingency, 1)
  same_class_pred = np.max(contingency, 0)
  diff_class_true = contingency.sum(axis=1) - same_class_true
  diff_class_pred = contingency.sum(axis=0) - same_class_pred
  total = contingency.sum()

  true_positives = (same_class_true * (same_class_true - 1)).sum()
  false_positives = (diff_class_true * same_class_true * 2).sum()
  false_negatives = (diff_class_pred * same_class_pred * 2).sum()
  true_negatives = total * (
      total - 1) - true_positives - false_positives - false_negatives

  return true_positives, false_positives, false_negatives, true_negatives

def pairwise_precision(y_true, y_pred):
  """Computes pairwise precision of two clusterings.
  Args:
    y_true: An [n] int ground-truth cluster vector.
    y_pred: An [n] int predicted cluster vector.
  Returns:
    Precision value computed from the true/false positives and negatives.
  """
  true_positives, false_positives, _, _ = _pairwise_confusion(y_true, y_pred)
  return true_positives / (true_positives + false_positives)


def pairwise_recall(y_true, y_pred):
  """Computes pairwise recall of two clusterings.
  Args:
    y_true: An (n,) int ground-truth cluster vector.
    y_pred: An (n,) int predicted cluster vector.
  Returns:
    Recall value computed from the true/false positives and negatives.
  """
  true_positives, _, false_negatives, _ = _pairwise_confusion(y_true, y_pred)
  return true_positives / (true_positives + false_negatives)


for t in range(1000):
    # dist = model_mlp(embs)
    # print(loss,C)

    dist = model_mlp1(0, graph.x, graph.edge_index, graph.edge_attr, graph.dist_row, degree, graph.x.shape[0])

    r = torch.softmax(10*dist, 1).to(device)
    bin_adj_nodiag = bin_adj.to(device) * (torch.ones(bin_adj.shape[0], bin_adj.shape[0]).to(device) - torch.eye(bin_adj.shape[0]).to(device))

    C = (math.sqrt(K)/bin_adj_nodiag.shape[0]) * (torch.norm(torch.sum(r.t(),dim=1),p=2)) - 1

    loss = -((1. / bin_adj_nodiag.sum()) * (r.t() @ mod @ r).trace())
    loss = loss.to(device)
    loss = loss + 10 * C

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    r0 = torch.softmax(100*r, dim=1).to(device)
    a = torch.argmax(r0, 1).to(device)

    # mooo = modularity(bin_adj_nodiag,a)
    # print("mooo",mooo)
    label = label.cpu()
    b = a.cpu()
    NMI = sklearn.metrics.normalized_mutual_info_score(label, b[label_indices], average_method='arithmetic')
    print("NMI",NMI)

    precision = pairwise_precision(label, b[label_indices])
    recall = pairwise_recall(label, b[label_indices])
    print('F1:', 2 * precision * recall / (precision + recall))

    cudu = conductance(bin_adj_nodiag,a)
    print("cudu",cudu)

    # 获取张量中的唯一值
    unique_values = torch.unique(a)
    # 使用 torch.bincount() 统计每个值的个数
    counts = torch.bincount(a)
    print(counts)
    S = bin_adj_nodiag.shape[0]/K
    BCI = sum(torch.abs(counts - S)/S)/K
    print("BCI:",BCI)

    internal_edges = calculate_internal_edges(bin_adj_nodiag, a)
    obj = (1. / bin_adj_nodiag.sum()) * (r0.t() @ mod_1 @ r0).trace()
    print(t, "-", internal_edges,obj)

    if(obj>loss_fin):
        loss_fin = obj


end = time.time()
print("time:", end - start)
print("obj:",loss_fin)
