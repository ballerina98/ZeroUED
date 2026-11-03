"""AE approach: train an encoder by making a decoder rebuild each signal from its features.
Never sees the test set."""

import numpy as np
import torch


class AETrainer:
    """Trains encoder + decoder together: noisy signal -> encoder -> features -> decoder -> rebuilt signal.
    The loss is how far the rebuilt signal is from the clean one."""

    def __init__(self, encoder, decoder, lr, batch_size, noise_std, seed, device):
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.batch_size = batch_size
        self.noise_std = noise_std
        self.device = device
        self.rng = np.random.default_rng(seed)                               # for shuffling
        self.optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), lr=lr)

    def train_epoch(self, x_train):
        """One pass over all training signals (numpy (N, 2, 256)), in a new random order. Returns the mean loss."""
        self.encoder.train()
        self.decoder.train()
        order = self.rng.permutation(len(x_train))                           # new order every epoch
        total = 0.0
        for start in range(0, len(order), self.batch_size):
            x = torch.from_numpy(x_train[order[start:start + self.batch_size]]).to(self.device)
            noisy = (x + torch.randn_like(x) * self.noise_std) / (1 + self.noise_std)   # as the old code
            rebuilt = self.decoder(self.encoder(noisy))
            loss = torch.mean((rebuilt - x.reshape(len(x), -1)) ** 2)        # rebuilt vs the CLEAN signal

            self.optimizer.zero_grad()      # forget the previous batch's nudges
            loss.backward()                 # work out how each number should change
            self.optimizer.step()           # apply the nudges
            total += loss.item() * len(x)
        return total / len(x_train)


if __name__ == "__main__":
    # Quick check on v1, fold 0: does the loss go down, and how long does one epoch take?
    import time
    from data.load_manysig import load_manysig
    from data.splits import make_folds
    from modality.raw_iq import raw_iq
    from feature_learning.kan import KANDecoder, KANEncoder

    signals, meta = load_manysig(equalized=True, receivers=["1-1"])
    x = raw_iq(signals, norm="max_abs")
    x_train = x[make_folds(meta, "v1")[0]["train"]]

    torch.manual_seed(0)                                                     # same starting weights every run
    encoder = KANEncoder(in_features=512, n_features=20, grid_size=10, spline_order=4)
    decoder = KANDecoder(n_features=20, out_features=512, grid_size=10, spline_order=3)
    trainer = AETrainer(encoder, decoder, lr=1e-3, batch_size=50, noise_std=0.1, seed=0, device="cpu")

    for epoch in range(3):
        start = time.time()
        loss = trainer.train_epoch(x_train)
        print(f"epoch {epoch + 1}: loss {loss:.5f}   {time.time() - start:.1f} s")