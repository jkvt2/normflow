import torch
from torch.utils.data import DataLoader

import numpy as np
from sklearn import datasets
from model import NFSequential, AffineCouple, Invertible1x1
import matplotlib.pyplot as plt
from collections import OrderedDict
plt.style.use('ggplot')

# Sample from our "dataset". We artificially have infinitely many data points here.
train_set = datasets.make_moons(n_samples=1000, noise=.05)[0].astype(np.float32)
test_set = datasets.make_moons(n_samples=100, noise=.05)[0].astype(np.float32)

dataLoader = DataLoader(
    dataset=train_set,
    batch_size=32,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
    drop_last = True)


torch.cuda.set_device(1)

dim = 2
model = NFSequential(OrderedDict([
           ('1x1_0', Invertible1x1(dim=dim, lu_decomposed=True)),
          ('afc_0', AffineCouple(dim=dim, flip=True)),
          ('1x1_1', Invertible1x1(dim=dim, lu_decomposed=True)),
          ('afc_1', AffineCouple(dim=dim, flip=False)),
          ('1x1_2', Invertible1x1(dim=dim, lu_decomposed=True)),
          ('afc_2', AffineCouple(dim=dim, flip=True)),
          ('1x1_3', Invertible1x1(dim=dim, lu_decomposed=True)),
          ('afc_3', AffineCouple(dim=dim, flip=False)),
          # ('afc_4', AffineCouple(dim=dim, flip=True)),
          # ('afc_5', AffineCouple(dim=dim, flip=False)),
          # ('afc_6', AffineCouple(dim=dim, flip=True)),
          # ('afc_7', AffineCouple(dim=dim, flip=False)),
        ]))

# Training hyperparameters.
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

p = torch.Tensor([[0, 1], [1, 0]])

# Iterate over the number of iterations.
for e in range(100):
    for i, batch in enumerate(dataLoader):
        total_step = e * len(dataLoader) + i
        # X = StandardScaler().fit_transform(X)
        
        optimizer.zero_grad()
        
        batch = torch.FloatTensor(batch).cuda()
        out_forwardpass = model.neglogprob(batch)
        loss = - torch.mean(out_forwardpass)
        
        # Backpropagation.
        loss.backward()
        optimizer.step()
        if total_step % 500 == 0:
            print('Iter {}, loss is {:.3f}'.format(total_step, loss.item()))
            # w = list(model.parameters())[0]
            # print('1x1_0.w', w.detach().cpu().flatten())
            
            # l, u = list(model.parameters())[:2]
            # l = l.detach().cpu()
            # u = u.detach().cpu()
            # w = torch.mm(torch.mm(p, l), u)
            # print('1x1_0.l', l.flatten())
            # print('1x1_0.u', u.flatten())
            # print('1x1_0.s', torch.diagonal(u))
            # print('  1x1_0.w', w.flatten())

new_Xs = model.sample_nvp_chain(N=10000, dim=dim)
new_Xs = new_Xs.data.cpu().numpy()

# Plot.
plt.scatter(new_Xs[:, 0], new_Xs[:, 1], c='r', s=1)
plt.show()