import math
from utils import *
import torch

from torch.nn.parameter import Parameter
import torch.nn as nn
from torch.nn.modules.module import Module
import torch.nn.functional as F

class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # kaiming_uniform(self.weight, fan=self.weight.size(1), a=math.sqrt(5))
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)
        #glorot(self.weight)
        #zeros(self.bias)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class MLP(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MLP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # kaiming_uniform(self.weight, fan=self.weight.size(1), a=math.sqrt(5))
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, inputs):
        output = torch.matmul(inputs, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GATconv(Module):
    def __init__(self, in_features, out_features, indices, concat=True, heads=1, negtive_slope=0.2, dropout=0, bias=True):
        super(GATconv, self).__init__()
        self.in_fetures = in_features
        self.out_features = out_features
        # self.indices = torch.as_tensor(indices).long()
        self.indices = indices
        self.heads = heads
        self.concat = concat
        self.negetive_slope = negtive_slope
        self.dropout = dropout

        self.weight = Parameter(torch.Tensor(in_features, heads * out_features))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_features))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_features))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, adj):
        x = torch.matmul(x, self.weight).view(-1, self.heads, self.out_features)
        alpha = (torch.cat([torch.index_select(x, 0, self.indices[0]), torch.index_select(x, 0, self.indices[1])], dim=-1) * self.att).sum(dim=-1)
        print(self.indices.size())
        print(adj.size())
        print(alpha.size())
        alpha = F.leaky_relu(alpha, self.negetive_slope)
        # alpha = softmax(alpha, self.indices[0])
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        attention_weights_matrix = torch.cat([torch.sparse.FloatTensor(self.indices, alpha_item, adj.shape) for alpha_item in alpha.transpose(1, 0)], dim=-1)
        print(attention_weights_matrix.size())
        alpha = softmax_sparse_tensor(attention_weights_matrix)

        # Sample attention coefficients stochastically.
        #alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        arr_out = x * alpha.view(-1, self.heads, 1)
        if self.concat is True:
            self.out = arr_out.view(-1, self.heads * self.out_features)
        else:
            self.out = arr_out.mean(dim=1)
        if self.bias is not None:
            self.out = self.out + self.bias

        return self.out

class GraphAttentionLayer(Module):
    def __init__(self, in_features, out_features, dropout, config, concat=True, bias=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.dropout = dropout
        self.config = config
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.a = Parameter(torch.Tensor(2 * out_features, 1))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.bias = self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        zeros(self.bias)

    def forward(self, x, adj):
        x = torch.matmul(x, self.weight)
        N = x.size()[0]
        a_input = torch.cat([x.repeat(1, N).view(N * N, -1), x.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(2), negative_slope=self.config.negative_slope)

        # zero_vec = -9e15*torch.ones_like(e)
        zero_vec = torch.zeros_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, x)
        return h_prime


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
