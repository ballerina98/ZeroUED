"""Evaluation: compare the detector's output with the truth. The only place labels are used."""

from sklearn.metrics import (accuracy_score, adjusted_mutual_info_score, f1_score,
                             normalized_mutual_info_score, precision_score, recall_score, roc_auc_score)


def evaluate(scores, clusters, is_unknown, tx, threshold):
    """Return the metrics of one fold as a dict.

    scores:     detector score per test signal (high = looks unknown)
    clusters:   nearest cluster per test signal
    is_unknown: True for test signals of the unknown transmitter (from meta)
    tx:         true transmitter name per test signal (from meta)
    threshold:  score above which a signal is called unknown (old code: 0.95)
    """
    predicted_unknown = scores > threshold
    return {
        "roc_auc": roc_auc_score(is_unknown, scores),                   # uses the scores, no threshold
        "f1": f1_score(is_unknown, predicted_unknown),                  # these four use the threshold
        "precision": precision_score(is_unknown, predicted_unknown),
        "recall": recall_score(is_unknown, predicted_unknown),
        "accuracy": accuracy_score(is_unknown, predicted_unknown),
        "ami": adjusted_mutual_info_score(tx, clusters),                # what the old code called "nmi"
        "nmi": normalized_mutual_info_score(tx, clusters),              # the real NMI
    }


if __name__ == "__main__":
    # Quick check on v1, fold 0 (14-10 unknown).
    from data.load_manysig import load_manysig
    from data.splits import make_folds
    from modality.raw_iq import raw_iq
    from feature_learning.pca import PCAEncoder
    from detection.kmeans import KMeansDetector

    signals, meta = load_manysig(equalized=True, receivers=["1-1"])
    x = raw_iq(signals, norm="max_abs")
    fold = make_folds(meta, "v1")[0]

    encoder = PCAEncoder(n_features=20).fit(x[fold["train"]])
    detector = KMeansDetector(n_clusters=80, seed=0).fit(encoder.encode(x[fold["train"]]))
    scores, clusters = detector.score(encoder.encode(x[fold["test"]]))

    test_tx = meta["tx"].iloc[fold["test"]].to_numpy()                  # the truth, used only here
    metrics = evaluate(scores, clusters, test_tx == fold["unknown"], test_tx, threshold=0.95)
    for name, value in metrics.items():
        print(f"{name:10s} {value:.3f}")