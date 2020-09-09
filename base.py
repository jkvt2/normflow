import torch
import numpy as np
from sklearn import datasets
from model import NF, AffineCouple, Invertible1x1
import matplotlib.pyplot as plt
from collections import OrderedDict
plt.style.use('ggplot')


torch.cuda.set_device(0)

dim = 2
model = NF(OrderedDict([
           ('1x1_0', Invertible1x1(dim=dim, lu_decomposed=True)),
          ('afc_0', AffineCouple(dim=dim, flip=True)),
           ('1x1_1', Invertible1x1(dim=dim, lu_decomposed=True)),
          ('afc_1', AffineCouple(dim=dim, flip=False)),
           ('1x1_2', Invertible1x1(dim=dim, lu_decomposed=True)),
          ('afc_2', AffineCouple(dim=dim, flip=True)),
           ('1x1_3', Invertible1x1(dim=dim, lu_decomposed=True)),
          ('afc_3', AffineCouple(dim=dim, flip=False)),
        ]))

# Training hyperparameters.
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Increase or decrease this if you wish.
iters = 5000

# Iterate over the number of iterations.
for i in range(iters):
    # Sample from our "dataset". We artificially have infinitely many data points here.
    noisy_moons = datasets.make_moons(n_samples=128, noise=.05)[0].astype(np.float32)
    # X = StandardScaler().fit_transform(X)
    
    optimizer.zero_grad()
    
    batch = torch.FloatTensor(noisy_moons).cuda()
    out_forwardpass = model.neglogprob(batch)
    loss = - torch.mean(out_forwardpass)
    
    # Backpropagation.
    loss.backward()
    optimizer.step()
    if i % 500 == 0:
        print('Iter {}, loss is {:.3f}'.format(i, loss.item()))
        print('1x1_0.weight', list(model.parameters())[0].detach().cpu().flatten())

new_Xs = model.sample_nvp_chain(N=10000, dim=dim)
new_Xs = new_Xs.data.cpu().numpy()

# Plot.
plt.scatter(new_Xs[:, 0], new_Xs[:, 1], c='r', s=1)
plt.show()