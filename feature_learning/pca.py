"""PCA feature learner: 20 numbers per signal, learned from the training signals. ML approach: none."""

### Note: while old PCA inherited something from CNN/KAN, it was left with 5 pretend epoch
# PCA doesn't learn gradually, it is calculated in exactly one step, so repeating it on the same signals will give same numbers
# For that reason - no need to separate v0 and v1, just polish that

### note to remember: here, PCA is doing 20 directions, probably to match CNN/KAN... but play with this later

from sklearn.decomposition import PCA

class PCAEncoder:
    # Learns the directions in which training signals differ most, then sedcribes any signal along them

    def __init__(self, n_features):
        self.pca = PCA(n_components=n_features, svd_solver="full")      # full - same result every time, exact

    def fit(self, x_train):
        # learn the directions from the training signals only
        self.pca.fit(x_train.reshape(len(x_train), -1))                 # (N, 2, 256) -> (N, 512)
        return self

    def encode(self, x):
        # describe signals with the learned direction; x: (N, 2, 256) -> (N, n_features)
        return self.pca.transform(x.reshape(len(x), -1))


if __name__ == "__main__":
    # Quick check on v1, fold 0 (14-10)
    from data.load_manysig import load_manysig
    from data.splits import make_folds
    from modality.raw_iq import raw_iq

    signals, meta = load_manysig(equalized=True, receivers=["1-1"])
    x = raw_iq(signals, norm="max_abs")
    fold = make_folds(meta, "v1")[0]

    # HERE IS 20 DIRECTIONS CHOICE
    encoder = PCAEncoder(n_features=20).fit(x[fold["train"]])        # learn from training signals only
    z_train = encoder.encode(x[fold["train"]])
    z_test = encoder.encode(x[fold["test"]])
    print("train", z_train.shape, " test", z_test.shape)
    print("share of variation kept by 20 directions:", round(encoder.pca.explained_variance_ratio_.sum(), 3))


