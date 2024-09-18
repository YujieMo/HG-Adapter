from sklearn.preprocessing import normalize
import scipy.sparse as sp
import torch as th
import torch.nn.functional as F
from dgl import function as fn
from sklearn.preprocessing import OneHotEncoder
import math
import dgl
from sklearn.metrics import normalized_mutual_info_score, pairwise, f1_score, accuracy_score
from dgl.base import DGLError
from dgl.utils import expand_as_pair
from torch import nn
from torch.nn import init
import torch
import os
import numpy as np
import random


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1.0).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_to_tuple(mx):
#     mx = normalize_adj(mx)
    if not sp.isspmatrix_coo(mx):
        mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col))
    values = mx.data
    shape = mx.shape
    return coords, values, shape


def preprocess_features(features, norm=True):
    """Row-normalize feature matrix and convert to tuple representation"""
    if sp.issparse(features):
        features = features.toarray()
    if norm:
        features[features>0] = 1
        # rowsum = np.array(features.sum(1))
        # r_inv = np.power(rowsum, -1.0).flatten()
        # r_inv[np.isinf(r_inv)] = 0.
        # r_mat_inv = sp.diags(r_inv)
        # features = r_mat_inv.dot(features)
    return th.FloatTensor(features)

def preprocess_features_freebase(features, norm=True):
    """Row-normalize feature matrix and convert to tuple representation"""
    if sp.issparse(features):
        features = features.toarray()
    if norm:
        features[features>0] = 1
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1.0).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
    return th.FloatTensor(features)


def normalize_mx(mx, diagonal=True):
    if diagonal:
        size = mx.shape[0]
        return normalize(mx+sp.eye(size), norm='l1', axis=1)
    else:
        return normalize(mx, norm='l1', axis=1)



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


def pairwise_distance(x, y=None):
    x = x.unsqueeze(0).permute(0, 2, 1)
    if y is None:
        y = x
    y = y.permute(0, 2, 1) # [B, N, f]
    A = -2 * th.bmm(y, x) # [B, N, N]
    A += th.sum(y**2, dim=2, keepdim=True) # [B, N, 1]
    A += th.sum(x**2, dim=1, keepdim=True) # [B, 1, N]
    return A.squeeze()

def printConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))
    print(args_names)
    print(args_vals)

def setup_seed(seed=0):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

import dgl.function as fn
class GCNConv_dgl(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCNConv_dgl, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, g):
        g.ndata['h'] = self.linear(x)
        g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h'))
        return g.ndata['h']


class GSL(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, nlayers, k, sparse):
        super(GSL, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hid_dim))
        for _ in range(nlayers - 2):
            self.layers.append(nn.Linear(hid_dim, hid_dim))
        self.layers.append(nn.Linear(hid_dim, out_dim))

        self.k = k
        self.sparse = sparse
        self.in_dim = in_dim
        self.mlp_knn_init()

    def mlp_knn_init(self):
        for layer in self.layers:
            layer.weight = nn.Parameter(torch.eye(self.in_dim))

    def forward(self, h, hom_new, args):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                h = F.relu(h)

        if self.sparse == 1:
            rows, cols, values = knn_fast(h, self.k, 1000) #h

            rows_ = torch.cat((rows, cols))
            cols_ = torch.cat((cols, rows))
            values_ = F.relu(torch.cat((values, values)))

            adj = dgl.graph((rows_, cols_), num_nodes=h.shape[0], device='cuda')
            adj.edata['w'] = values_
        else:
            embeddings = F.normalize(h, dim=1, p=2) #h
            adj = torch.mm(embeddings, embeddings.t())

            adj = top_k(adj, self.k + 1, args)
            adj = F.relu(adj)

        return adj

class GCNConv_dense(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCNConv_dense, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def init_para(self):
        self.linear.reset_parameters()

    def forward(self, x, adj):

        x = self.linear(x)
        x = torch.matmul(adj, x)

        return x

class GCN_DAE(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, nlayers, dropout_cla, dropout_adj, mlp_dim, k, sparse):
        super(GCN_DAE, self).__init__()

        self.layers = nn.ModuleList()
        if sparse == 1:
            self.layers.append(GCNConv_dgl(in_dim, hid_dim))
            for i in range(nlayers - 2):
                self.layers.append(GCNConv_dgl(hid_dim, hid_dim))
            self.layers.append(GCNConv_dgl(hid_dim, out_dim))

            self.dropout_cla = dropout_cla
            self.dropout_adj = dropout_adj

        else:
            self.layers = nn.ModuleList()
            self.layers.append(GCNConv_dense(in_dim, hid_dim))
            for i in range(nlayers - 2):
                self.layers.append(GCNConv_dense(hid_dim, hid_dim))
            self.layers.append(GCNConv_dense(hid_dim, out_dim))

            self.dropout_cla = dropout_cla
            self.dropout_adj = nn.Dropout(p=dropout_adj)

        self.sparse = sparse
        self.graph_generator = GSL(in_dim, math.floor(math.sqrt(in_dim * mlp_dim)), mlp_dim, nlayers, k, sparse)
        # self.graph_generator = GSL(hid_dim, hid_dim, hid_dim, nlayers, k, sparse)

    def get_adj(self, features, hom_new, args):
        if self.sparse == 1:
            return self.graph_generator(features, hom_new, args)
        else:
            adj = self.graph_generator(features,hom_new, args)
            adj = (adj + adj.T) / 2

            inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + 1e-10)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]

    def forward(self, features, x, hom_new, args):
        adj = self.get_adj(features, hom_new, args)  # 由GSL产生adj
        if self.sparse == 1:
            adj_dropout = adj
            adj_dropout.edata['w'] = F.dropout(adj_dropout.edata['w'], p=self.dropout_adj, training=self.training)
        else:
            adj_dropout = self.dropout_adj(adj)

        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, adj_dropout)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_cla, training=self.training)
        x = self.layers[-1](x, adj_dropout)

        return x, adj

    def get_loss_homophily(self, g, logits, labels, train_mask, nclasses, num_hop, sparse, args):

        logits = torch.argmax(logits, dim=-1, keepdim=True)
        logits[train_mask, 0] = labels[train_mask]

        preds = torch.zeros(logits.shape[0], nclasses).to(args.device)
        preds = preds.scatter(1, logits, 1).detach()

        if sparse == 1:
            g.ndata['l'] = preds
            for _ in range(num_hop):
                g.update_all(fn.u_mul_e('l', 'w', 'm'), fn.sum(msg='m', out='l'))
            q_dist = F.log_softmax(g.ndata['l'], dim=-1)
        else:
            q_dist = preds
            for _ in range(num_hop):
                q_dist = torch.matmul(g, q_dist)
            q_dist = F.log_softmax(q_dist, dim=-1)

        loss_hom = F.kl_div(q_dist, preds)

        return loss_hom

def knn_fast(X, k, b):

    X = torch.nn.functional.normalize(X, dim=1, p=2)

    index = 0
    values = torch.zeros(X.shape[0] * (k + 1)).cuda()
    rows = torch.zeros(X.shape[0] * (k + 1)).cuda()
    cols = torch.zeros(X.shape[0] * (k + 1)).cuda()
    norm_row = torch.zeros(X.shape[0]).cuda()
    norm_col = torch.zeros(X.shape[0]).cuda()

    while index < X.shape[0]:
        if (index + b) > (X.shape[0]):
            end = X.shape[0]
        else:
            end = index + b

        sub_tensor = X[index:index + b]
        similarities = torch.mm(sub_tensor, X.t())
        vals, inds = similarities.topk(k=k + 1, dim=-1)

        values[index * (k + 1):(end) * (k + 1)] = vals.view(-1)
        cols[index * (k + 1):(end) * (k + 1)] = inds.view(-1)
        rows[index * (k + 1):(end) * (k + 1)] = torch.arange(index, end).view(-1, 1).repeat(1, k + 1).view(-1)
        norm_row[index: end] = torch.sum(vals, dim=1)
        norm_col.index_add_(-1, inds.view(-1), vals.view(-1))
        index += b

    norm = norm_row + norm_col
    rows = rows.long()
    cols = cols.long()
    values *= (torch.pow(norm[rows], -0.5) * torch.pow(norm[cols], -0.5))

    return rows, cols, values

def top_k(raw_graph, k, args):
    _, indices = raw_graph.topk(k=int(k), dim=-1)

    mask = torch.zeros(raw_graph.shape).to(args.device)
    mask[torch.arange(raw_graph.shape[0]).view(-1, 1), indices] = 1.

    mask.requires_grad = False
    sparse_graph = raw_graph * mask

    return sparse_graph

def get_random_mask(features, r, scale, args):

    if args.dataset == 'ogbn-arxiv' or args.dataset == 'minist' or args.dataset == 'cifar10' or args.dataset == 'fashionmnist':
        probs = torch.full(features.shape, 1 / r)
        mask = torch.bernoulli(probs)
        return mask

    nones = torch.sum(features > 0.0).float()
    nzeros = features.shape[0] * features.shape[1] - nones
    pzeros = nones / nzeros / r * scale

    probs = torch.zeros(features.shape).to(args.device)
    probs[features == 0.0] = pzeros
    probs[features > 0.0] = 1 / r

    mask = torch.bernoulli(probs)

    return mask

def get_loss_reconstruction(model, features, mask, hom_new, dataset, args):

    if dataset == 'ogbn-arxiv' or dataset == 'minist' or dataset == 'cifar10' or dataset == 'fashionmnist':
        masked_features = features * (1 - mask)
        logits, adj = model(features, masked_features)

        indices = mask > 0
        loss = F.mse_loss(logits[indices], features[indices], reduction='mean')

        return loss, adj

    logits, adj = model(features, features, hom_new, args)

    indices = mask > 0
    loss = F.binary_cross_entropy_with_logits(logits[indices], features[indices], reduction='mean')

    return loss, adj


class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, args):
        super(GCN, self).__init__()

        self.n_layers = n_layers
        self.convs = nn.ModuleList()

        self.convs.append(GraphConv(in_dim, hid_dim, norm='both'))

        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GraphConv(hid_dim, hid_dim, norm='both'))
            self.convs.append(GraphConv(hid_dim, out_dim, norm='both'))

    def forward(self, graph, x):

        for i in range(self.n_layers - 1):
            x = F.relu(self.convs[i](graph, x))
        x = self.convs[-1](graph, x)

        return x


class GraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 weight=True,
                 bias=True,
                 activation=None,
                 allow_zero_in_degree=False):
        super(GraphConv, self).__init__()

        if norm not in ('none', 'both', 'right', 'left'):
            raise DGLError('Invalid norm value. Must be either "none", "both", "right" or "left".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, weight=None, edge_weight=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            aggregate_fn = fn.copy_u('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm in ['left', 'both']:
                degs = graph.out_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = th.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError('External weight is provided while at the same time the'
                                   ' module has defined its own weight parameter. Please'
                                   ' create the module with flag weight=False.')
            else:
                weight = self.weight

            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    feat_src = th.matmul(feat_src, weight)
                graph.srcdata['h'] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
            else:
                # aggregate first then mult W
                graph.srcdata['h'] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
                if weight is not None:
                    rst = th.matmul(rst, weight)

            if self._norm in ['right', 'both']:
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst

    def extra_repr(self):
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)

def random_aug(x, feat_drop_rate):
    feat = drop_feature(x, feat_drop_rate)

    return feat

def drop_feature(x, drop_prob):
    drop_mask = th.empty(
        (x.size(1),),
        dtype=th.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

def mask_edge(graph, mask_prob):
    E = graph.number_of_edges()

    mask_rates = th.FloatTensor(np.ones(E) * mask_prob)
    masks = th.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx

def GNNNodeEva(out, prompt, labels, idx):
    from sklearn.metrics import f1_score
    prompt.eval()
    pred = out.argmax(dim=1)
    f1_macro = f1_score(torch.argmax(labels[idx], dim=1).cpu(), pred[idx].cpu(), average='macro')
    f1_micro = f1_score(torch.argmax(labels[idx], dim=1).cpu(), pred[idx].cpu(), average='micro')
    return f1_macro, f1_micro

def pairwise_distance(x, y=None):
    x = x.squeeze().unsqueeze(0).permute(0, 2, 1)
    if y is None:
        y = x
    y = y.permute(0, 2, 1)  # [B, N, f]
    A = -2 * torch.bmm(y, x)  # [B, N, N]
    A += torch.sum(y ** 2, dim=2, keepdim=True)  # [B, N, 1]
    A += torch.sum(x ** 2, dim=1, keepdim=True)  # [B, 1, N]
    return A.squeeze()


def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot


class MLP(nn.Module):
    def __init__(self, dim, dropprob=0.0):
        super(MLP, self).__init__()
        self.net = nn.ModuleList()
        self.dropout = torch.nn.Dropout(dropprob)
        struc = []
        for i in range(len(dim)):
            struc.append(dim[i])
        for i in range(len(struc) - 1):
            self.net.append(nn.Linear(struc[i], struc[i + 1]))

    def forward(self, x):
        for i in range(len(self.net) - 1):
            x = F.relu(self.net[i](x))
            x = self.dropout(x)
        y = self.net[-1](x)

        return y