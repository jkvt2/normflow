from torch import distributions
import torch
import torch.nn as nn
import numpy as np

from scipy.stats import special_ortho_group
from scipy.linalg import lu

class Invertible1x1(nn.Module):
    def __init__(self, dim, lu_decomposed=True):
        super().__init__()
        # w = special_ortho_group.rvs(dim)
        # print(w)
        w = np.array([[-0.47315506, 0.88097916],
                      [-0.88097916, -0.47315506]])
        if lu_decomposed:
            p, l, u = lu(w)
            self.p = torch.from_numpy(p).float().cuda()
            self.l = nn.Parameter(torch.from_numpy(l).float().cuda())
            self.u = nn.Parameter(torch.from_numpy(u).float().cuda())
        else:
            self.w = nn.Parameter(torch.from_numpy(w).float().cuda())
        self.lu_decomposed = lu_decomposed
    
    @staticmethod
    def compose_w(p, l, u):
        l = torch.tril(l)
        u = torch.triu(u)
        return torch.mm(torch.mm(p, l), u)
  
    def forward(self, x):
        if self.lu_decomposed:
            w = self.compose_w(self.p, self.l, self.u)
        else:
            w = self.w
        y = torch.mm(x, w)
        return y
    
    def invert(self, y):
        if self.lu_decomposed:
            w = self.compose_w(self.p, self.l, self.u)
            log_det = torch.sum(torch.log(
                torch.abs(torch.diagonal(self.u))))
        else:
            w = self.w
            log_det = torch.log(
                torch.abs(torch.det(w)))
        x = torch.mm(y, torch.inverse(w))
        return x, log_det.expand(x.shape[0])

class AffineCouple(nn.Module):
    def __init__(self, dim, flip, n_features=256, n_layers=3, activation=nn.ReLU):
        super().__init__()
        assert n_layers >= 2
        layers = [nn.Linear(dim//2, n_features), activation(),] + \
            [nn.Linear(n_features, n_features), activation(),] * (n_layers-2) + \
            [nn.Linear(n_features, dim)]
        self.shift_log_scale_fn = nn.Sequential(*layers).cuda()
        self.dim = dim
        self.flip = flip
    
    def forward(self, x):
        # x is of shape [B, H]
        d = x.shape[-1] // 2
        x1, x2 = x[:, :d], x[:, d:]
        if self.flip:
            x2, x1 = x1, x2
        net_out = self.shift_log_scale_fn(x1)
        shift = net_out[:, :self.dim // 2]
        log_scale = net_out[:, self.dim // 2:]
        y2 = x2 * torch.exp(log_scale) + shift
        if self.flip:
            x1, y2 = y2, x1
        y = torch.cat([x1, y2], -1)
        return y

    def invert(self, y):
        d = y.shape[-1] // 2
        y1, y2 = y[:, :d], y[:, d:]
        if self.flip:
            y1, y2 = y2, y1
        net_out = self.shift_log_scale_fn(y1)
        shift = net_out[:, :self.dim // 2]
        log_scale = net_out[:, self.dim // 2:]
        x2 = (y2 - shift) * torch.exp(-log_scale)
        if self.flip:
            y1, x2 = x2, y1
        x = torch.cat([y1, x2], -1)
        log_det = log_scale.sum(-1)
        return x, log_det

class NFSequential(nn.Sequential):
    @staticmethod
    def base_log_prob_fn(x):
        return torch.sum(- (x ** 2) / 2 - np.log(np.sqrt(2 * np.pi)), -1)
    
    def base_sample_fn(self, N, dim):
        # sampler random normal(0, I)
        prior = distributions.MultivariateNormal(torch.zeros(dim), torch.eye(dim))
        x = prior.sample((N, 1)).cuda().squeeze(1)
        return x
        
    def sample_nvp_chain(self, N, dim):
        x = self.base_sample_fn(N, dim)
        return self(x)
    
    def invert(self, y):
        for module in reversed(self):
            y, _ = module.invert(y=y)
        return y
    
    def neglogprob(self, y):
        # Run y through all the necessary inverses, keeping track
        # of the logscale along the way, allowing us to compute the loss.
        logscales = y.data.new(y.shape[0]).zero_()
        for module in reversed(self):
            y, logscale = module.invert(y=y)
            # One logscale per element in a batch per layer of flow.
            logscales += logscale
        return self.base_log_prob_fn(y) - logscales