import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
import os
import create_more
from torch_geometric.datasets import Planetoid
import torch

import numpy as np
import torch
import pickle
import time

# 定义随机块模型参数
block_sizes = [200,200,200,200,200]
# edge_probs = [[0.2,0.01,0.01,0.01,0.01,0.01,0.01,0.01],
#               [0.01,0.2,0.01,0.01,0.01,0.01,0.01,0.01],
#               [0.01,0.01,0.2,0.01,0.01,0.01,0.01,0.01],
#               [0.01,0.01,0.01,0.2,0.01,0.01,0.01,0.01],
#               [0.01,0.01,0.01,0.01,0.2,0.01,0.01,0.01],
#               [0.01,0.01,0.01,0.01,0.01,0.2,0.01,0.01],
#               [0.01,0.01,0.01,0.01,0.01,0.01,0.2,0.01],
#               [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.2]]
# edge_probs = [[0.2,0.01,0.01,0.01,0.01],
#               [0.01,0.2,0.01,0.01,0.01],
#               [0.01,0.01,0.2,0.01,0.01],
#               [0.01,0.01,0.01,0.2,0.01],
#               [0.01,0.01,0.01,0.01,0.2]]
edge_probs = [
    [0.2, 0.05, 0.05, 0.05, 0.05],
    [0.05, 0.2, 0.05, 0.05, 0.05],
    [0.05, 0.05, 0.2, 0.05, 0.05],
    [0.05, 0.05, 0.05, 0.2, 0.05],
    [0.05, 0.05, 0.05, 0.05, 0.2]
]
# 创建随机块模型数据集
dataset = torch_geometric.datasets.StochasticBlockModelDataset(
    root='.',  # 存储数据的根目录
    block_sizes=block_sizes,
    edge_probs=edge_probs,
    num_channels=256
)

# 获取生成的图
data = dataset[0]
print("1")
dataset_cora = Planetoid(root='./cora/', name='Cora')
# 打印数据集
print(dataset_cora)

# 提取data，并转换为device格式
data_cora = dataset_cora[0]
data = data_cora
indices = np.arange(0, data_cora.y.shape[0])
label = data_cora.y[(data_cora.train_mask == 1) | (data_cora.test_mask == 1) | (data_cora.val_mask == 1)]
label_indices = indices[(data_cora.train_mask == 1) | (data_cora.test_mask == 1) | (data_cora.val_mask == 1)]


device = torch.device('cuda:1')


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

features = data.x
labels = data.y
adj_all = to_dense_adj(data.edge_index).squeeze()
# adj_all, features, labels = load_data("./data/cora/","cora")
# features = data_cora.x
# labels = data_cora.y
# adj_all = to_dense_adj(data_cora.edge_index).squeeze()


n = adj_all.shape[0]
n_nodes=adj_all.shape[0]
bin_adj = adj_all.float()
dist_all = make_all_dists(bin_adj, 100)
diameter = dist_all[dist_all < 100].max()
# dist_all[dist_all == 100] = diameter*5
edge_index_all, edge_weight_all = torch_geometric.utils.dense_to_sparse(bin_adj)
# edge_weight_all = edge_weight_all.reshape((edge_weight_all.shape[0],-1))
dist_row = torch.sum(dist_all, dim=1)
dist_row = dist_row.unsqueeze(1)
diameter = dist_all.max()
degree = torch.sum(bin_adj, dim=0).to(device)
degree = degree.long()

graph = torch_geometric.data.Data(x=features.to(device), edge_index=edge_index_all.to(device), edge_attr=edge_weight_all.to(device), dist_row=dist_row.to(device), degree=degree.to(device))

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_nodes):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

        self.be = nn.BatchNorm1d(out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.bd = nn.BatchNorm1d(out_channels)

        #1
        self.fc_0 = nn.Linear(1,out_channels)
        self.fc_1 = nn.Embedding(2000, out_channels)
        self.fc_2 = nn.Linear(3*out_channels, out_channels)

    # 1
    def forward(self, x, edge_index, edge_weight, edges, degree):
        # print(edges.dtype, degree.dtype)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, 0.2)
        x = self.conv2(x, edge_index, edge_weight)
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

    def forward(self, idx, x, edge_index, edge_weight, edge, degree, batch, dist, perm):
        # Compute query embeddings
        embs_q = self.q_net(x, edge_index, edge_weight, edge, degree)
        embs_q = F.normalize(embs_q, dim=1)
        # q = F.normalize(q, dim=1)
        q = embs_q[idx*batch:(idx+1)*batch,:] # n * c

        w = dist[idx*batch:(idx+1)*batch,:]
        p = perm[idx*batch:(idx+1)*batch]

        weight = self.queue_w.clone().detach()[p]
        weight = weight / diameter * 16 - 6
        weight = torch.sigmoid(weight)
        # Compute key embeddings
        with torch.no_grad():
            self._momentum_update_key_encoder()
            embs_k = self.k_net(x, edge_index, edge_weight, edge, degree)
            embs_k = F.normalize(embs_k, dim=1)
            k = embs_k[idx * batch:(idx + 1) * batch, :]  # n * c

        # Positive pairs
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # l_pos = l_pos * 0.004

        # Negative pairs
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        l_neg = l_neg * weight

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= 0.01


        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, w)

        return embs_q, logits, labels

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.q_net.parameters(), self.k_net.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, weight):
        # gather keys before updating queue
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        self.queue_w[:, ptr: ptr + batch_size] = weight.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr


import torch.optim as optim
# train_dataset = create_more.get_random_data(n_nodes, 2, 0, device)
# train_dataset = create_more.get_starbucks_data(device)

model = MocoModel(graph.x.shape[1], 128, 64, n_nodes).to(device)
optimizer = optim.Adam(model.q_net.parameters(),lr=0.001, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss().to(device)

def train(x, edge_index, edge_weight, dist_row, degree, batch_size,n_nodes,dist,perm):
    loss_list = []
    model.train()
    num_batches = (n_nodes // batch_size) # 舍去最后的余数
    for i in range(num_batches):
        embs, logits, labels = model(i, x, edge_index, edge_weight, dist_row, degree, batch_size, dist,perm)
        loss = criterion(logits, labels)
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss)
    loss_avg = sum(loss_list) / len(loss_list)
    return loss_avg.item()

loss_fin = 100
for epoch in range(200):
    n_nodes = graph.x.shape[0]
    perm = torch.arange(n_nodes)
    loss = train(graph.x, graph.edge_index, graph.edge_attr, graph.dist_row, degree,16, n_nodes, dist_all,perm)
    print('Epoch {}, loss {:.4f}'.format(epoch, loss))
    perm = torch.randperm(features.shape[0])
    # 用新的节点编号对x和edge_index进行重新编号
    perm = torch.randperm(n_nodes)
    graph.x = graph.x[perm]
    dist_all = dist_all[perm]
    graph.edge_index[0] = perm[graph.edge_index[0]]
    graph.edge_index[1] = perm[graph.edge_index[1]]
    if (epoch > 100 and loss_fin > loss):
        loss_fin = loss
        torch.save(model.state_dict(), 'pre.pth')