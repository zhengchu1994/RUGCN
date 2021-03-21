from layers import *
from utils import *
import numpy as np


class GaussLayer(nn.Module):
    def __init__(self, nfeat, nhid, dim, nclass, config, dropout=0., alpha=0.1, device= 'cpu'):
        super(GaussLayer, self).__init__()
        self.alpha = alpha
        self.hiddenLayer = GraphConvolution(nfeat, nhid)
        self.muLayer = GraphConvolution(nhid, dim * nclass)
        self.sigmaLayer = GraphConvolution(nhid, dim * nclass)
        self.dropout = dropout
        self.n_class = nclass
        self.dim = dim
        self.config = config
        self.dim_list = [dim] * nclass
        self.device = device
        self.pattern = torch.as_tensor(np.split(np.random.permutation(range(dim * nclass)), nclass)).long().to(device)
        self.reset_parameters()
        # self.a = Parameter(torch.Tensor([0]))
    def reset_parameters(self):
        self.hiddenLayer.reset_parameters()
        self.muLayer.reset_parameters()
        self.sigmaLayer.reset_parameters()

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        if self.config.use_relu:
            self.x = self.hiddenLayer(x, adj)
        else:
            self.x = F.leaky_relu(self.hiddenLayer(x, adj), negative_slope=self.config.negative_slope)

        self.mu = self.muLayer(self.x, adj)
        self.sigma = F.elu(self.sigmaLayer(self.x, adj)) + 1 + 1e-14

        if self.dim != 1:
           self.logits = F.log_softmax(torch.stack([torch.index_select(self.mu, 1, index) for index in self.pattern]).mean(2).transpose(0, 1), dim=1)
        else:
            self.logits = F.log_softmax(self.mu, dim=1)
        return self.mu, self.sigma, self.logits

    def Wloss(self, edges):
        WD = Wasserstein(self.mu, self.sigma, edges[:, 0], edges[:, 1])
        return torch.mean(WD)


    def KLloss(self, edges, probs):
        KL = KLDivergence(self.mu, self.sigma, edges[:, 0], edges[:, 1])
        # KL = torch.exp(KL)
        return torch.mean(KL * probs)

    def closs(self, mask_x, mask_y):
        return F.nll_loss(self.logits[mask_x], mask_y)

    def build_loss2(self, edges, probs, mask_x, mask_y):
        return self.alpha * self.KLloss(edges, probs) + self.closs(mask_x, mask_y)

    def KLloss_old(self, edges):
        KL = KLDivergence(self.mu, self.sigma, edges[:, 0], edges[:, 1])
        # KL = torch.exp(KL)
        return torch.mean(KL)


    def build_loss(self, edges, mask_x, mask_y):
        return self.alpha * self.KLloss_old(edges) + self.closs(mask_x, mask_y)










