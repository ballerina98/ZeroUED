"""Detection: k-means on the training features; a test signal far out in its nearest cluster looks unknown."""


import numpy as np
from sklearn.cluster import KMeans

class KMeansDetector: 
    # Groups the training features into clusters, then scores how far out each test signal lies

    def __init__(self, n_clusters, seed):
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)

    def fit(self, z_train):
        # Cluster the training features; remember, per cluster, how far its members are from the center
        dist=self.kmeans.fit_transform(z_train)             # (N, n_clusters):distance
        cluster=dist.argmin(axis=1)                         # each sugnal's own (nearest) cluster
        self.member_dist=[np.sort(dist[cluster==c, c]) for c in range (dist.shape[1])]
        return self

    def score(self, z_test):
        # Per test signal: the fraction of its nearest cluster's training members that are closer to the center
        # 0.99 = farther out than 99% of them; high=likely known; Also returns the nearest cluster

        dist = self.kmeans.transform(z_test)
        cluster = dist.argmin(axis=1)
        d = dist[np.arange(len(z_test)), cluster]
        scores = np.ones(len(z_test))                       # an empty cluster counts as "far out" (as the old code)
        for i, (c, di) in enumerate(zip(cluster, d)):
            if len(self.member_dist[c]):
                scores[i] = np.searchsorted(self.member_dist[c], di) / len(self.member_dist[c])
        return scores, cluster

if __name__ == "__main__":
    # Quick check on v1, fold 0 (14-10 unknown): do unknown test signals get higher scores?
    from data.load_manysig import load_manysig
    from data.splits import make_folds
    from modality.raw_iq import raw_iq
    from feature_learning.pca import PCAEncoder

    signals, meta = load_manysig(equalized=True, receivers=["1-1"])
    x = raw_iq(signals, norm="max_abs")
    fold = make_folds(meta, "v1")[0]

    encoder = PCAEncoder(n_features=20).fit(x[fold["train"]])
    detector = KMeansDetector(n_clusters=80, seed=0).fit(encoder.encode(x[fold["train"]]))
    scores, _ = detector.score(encoder.encode(x[fold["test"]]))

    # Labels only for checking, never for fitting:
    is_unknown = (meta["tx"].iloc[fold["test"]] == fold["unknown"]).to_numpy()
    for name, group in [("known", ~is_unknown), ("unknown", is_unknown)]:
        print(f"{name:8s} mean score {scores[group].mean():.3f}   above 0.95: {(scores[group] > 0.95).mean():.1%}")           

