"""CL approach (contrastive learning, SimCLR-style): two altered copies (views) of every signal in the batch;
the encoder + head are trained so each view finds its partner among all the other views. Never sees the test set.
Reproduces the old code, quirks included (marked OLD QUIRK), so we can later fix them one at a time."""

import numpy as np
import torch
import torch.nn.functional as F

KINDS = ["noise", "bias", "amplifier", "rotation"]


def augment(x, kind, rng, std=0.05, n_chunks=4, prob=0.5):
    """One altered copy of a batch x (B, 2, 256), as the old code: the signal is cut into 4 chunks of 64 samples,
    and each chunk is changed with probability 0.5 (the same chunks for the whole batch)."""
    out = x.clone()
    size = x.shape[-1] // n_chunks
    for c in range(n_chunks):
        if rng.random() >= prob:                                             # this chunk stays as it is
            continue
        s = slice(c * size, (c + 1) * size)
        part = x[..., s]
        if kind == "noise":                                                  # every number + a small random number
            out[..., s] = part + torch.randn_like(part) * std
        elif kind == "bias":                                                 # one small number added to the chunk
            out[..., s] = part + torch.randn(1, device=x.device) * std
        elif kind == "amplifier":                                            # every number x (1 + small random)
            out[..., s] = part * (torch.randn_like(part) * std + 1)
        elif kind == "rotation":                                             # I/Q turned by one small angle
            g = torch.randn(1, device=x.device) * std
            out[..., 0, s] = part[..., 0, :] * torch.cos(g) - part[..., 1, :] * torch.sin(g)
            out[..., 1, s] = part[..., 0, :] * torch.sin(g) + part[..., 1, :] * torch.cos(g)
    return out


def contrastive_loss(u_p, u_q, temperature):
    """InfoNCE as the old code. Each of the 2B views: its partner vs the other 2B - 2 views. Returns 1 number."""
    p, q = F.normalize(u_p, dim=1), F.normalize(u_q, dim=1)                 # length 1, so p @ q.T = cos
    n = len(p)
    pq, pp, qq = p @ q.T, p @ p.T, q @ q.T                                   # (B, B) each; pq diagonal = partners
    others = ~torch.eye(n, dtype=torch.bool, device=p.device)
    pp, qq = pp[others].view(n, n - 1), qq[others].view(n, n - 1)           # drop each view's similarity to itself
    partner = torch.arange(n, device=p.device)                               # column i of pq holds i's partner
    loss_p = F.cross_entropy(torch.cat([pq, pp], 1) / temperature, partner) # B rows of 2B - 1 sims -> 1 average
    loss_q = F.cross_entropy(torch.cat([pq, qq], 1) / temperature, partner) # OLD QUIRK: should be pq.T
    return (loss_p + loss_q) / 2


class CLTrainer:
    """Encoder + projection head (20 -> 20, ReLU, 20 -> 20), trained together; the head is thrown away after."""

    def __init__(self, encoder, n_features, lr, batch_size, temperature, aug_std, seed, device):
        self.encoder = encoder.to(device)
        self.head = torch.nn.Sequential(torch.nn.Linear(n_features, n_features), torch.nn.ReLU(),
                                        torch.nn.Linear(n_features, n_features)).to(device)
        self.batch_size = batch_size
        self.temperature = temperature
        self.aug_std = aug_std
        self.device = device
        self.rng = np.random.default_rng(seed)                               # shuffling and augmentation choices
        self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.head.parameters()), lr=lr)

    def train_epoch(self, x_train):
        """One pass over all training signals (numpy (N, 2, 256)), in a new random order. Returns the mean loss."""
        self.encoder.train()
        self.head.train()
        order = self.rng.permutation(len(x_train))
        total = 0.0
        for start in range(0, len(order), self.batch_size):
            x = torch.from_numpy(x_train[order[start:start + self.batch_size]]).to(self.device)
            kind_p, kind_q = self.rng.choice(KINDS, size=2)                 # 2 of the 4 kinds (can be the same)
            x_p = augment(x, kind_p, self.rng, self.aug_std)                 # B views p
            base_q = x if kind_p == "rotation" else x_p                      # OLD QUIRK: noise/bias/amplifier
            x_q = augment(base_q, kind_q, self.rng, self.aug_std)            #   changed x in place, so q starts from p
            u_p = self.head(self.encoder(x_p))                               # (B, 20)
            u_q = self.head(self.encoder(x_q))                               # (B, 20)
            loss = contrastive_loss(u_p, u_q, self.temperature)

            self.optimizer.zero_grad()      # forget the previous batch's nudges
            loss.backward()                 # work out how each number should change
            self.optimizer.step()           # apply the nudges
            total += loss.item() * len(x)
        return total / len(x_train)


if __name__ == "__main__":
    # Quick check on v1, fold 0: the loss starts near ln(255) = 5.54 and, with temperature 1, cannot go below 3.57.
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
    trainer = CLTrainer(encoder, n_features=20, lr=1e-4, batch_size=128, temperature=1.0, aug_std=0.05,
                        seed=0, device="cpu")

    for epoch in range(2):
        start = time.time()
        loss = trainer.train_epoch(x_train)
        print(f"epoch {epoch + 1}: loss {loss:.4f}   {time.time() - start:.1f} s")