"""DC approach (deep clustering): every few epochs, K-means on the features of all training signals gives each
signal a group; in between, the encoder is trained to predict its signal's group. Never sees the test set."""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


class DCTrainer:
    """As the old code: no head. The encoder's 20 features are used directly as the scores of 20 groups."""

    def __init__(self, encoder, n_groups, update_every, lr, batch_size, seed, device):
        self.encoder = encoder.to(device)
        self.n_groups = n_groups
        self.update_every = update_every                                     # new groups every this many epochs
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self.rng = np.random.default_rng(seed)                               # for shuffling
        self.optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
        self.epoch = 0
        self.groups = None                                                   # one group number per training signal

    def update_groups(self, x_train):
        """Step 0: K-means on the features of ALL training signals, gradient off. These are the new answers."""
        z = self.encoder.encode(x_train)                                     # (N, 20), no gradient
        kmeans = KMeans(n_clusters=self.n_groups, n_init=10, random_state=self.seed)
        self.groups = torch.from_numpy(kmeans.fit_predict(z)).long()          # (N,), same order as x_train

    def train_epoch(self, x_train):
        """One pass over all training signals (numpy (N, 2, 256)), in a new random order. Returns the mean loss."""
        if self.epoch % self.update_every == 0:                              # before epochs 1, 21, 41, ...
            self.update_groups(x_train)
        self.epoch += 1
        self.encoder.train()                                                 # encode() had switched it to eval
        order = self.rng.permutation(len(x_train))                           # new order every epoch
        total = 0.0
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]                       # which signals are in this batch
            x = torch.from_numpy(x_train[idx]).to(self.device)
            scores = self.encoder(x)                                         # (B, 20): the features ARE the scores
            loss = F.cross_entropy(scores, self.groups[idx].to(self.device)) # B losses -> 1 average

            self.optimizer.zero_grad()      # forget the previous batch's nudges
            loss.backward()                 # work out how each number should change
            self.optimizer.step()           # apply the nudges
            total += loss.item() * len(x)
        return total / len(x_train)


if __name__ == "__main__":
    # Quick check on v1, fold 0: does the loss go down, and how big are the groups?
    import time
    from data.load_manysig import load_manysig
    from data.splits import make_folds
    from modality.raw_iq import raw_iq
    from feature_learning.kan import KANEncoder

    signals, meta = load_manysig(equalized=True, receivers=["1-1"])
    x = raw_iq(signals, norm="max_abs")
    x_train = x[make_folds(meta, "v1")[0]["train"]]

    torch.manual_seed(0)                                                     # same starting weights every run
    encoder = KANEncoder(in_features=512, n_features=20, grid_size=10, spline_order=4)
    trainer = DCTrainer(encoder, n_groups=20, update_every=20, lr=1e-3, batch_size=50, seed=0, device="cpu")

    for epoch in range(2):
        start = time.time()
        loss = trainer.train_epoch(x_train)
        print(f"epoch {epoch + 1}: loss {loss:.4f}   {time.time() - start:.1f} s")
    print("signals per group:", torch.bincount(trainer.groups, minlength=20).tolist())