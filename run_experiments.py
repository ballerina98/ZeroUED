"""Run one experiment from its settings file: python run_experiments.py experiments/manysig_v1_pca.yaml"""

import shutil
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import torch
import yaml

from data.load_manysig import load_manysig
from data.splits import make_folds
from detection.kmeans import KMeansDetector
from evaluation.metrics import evaluate
from feature_learning.cnn import CNNEncoder
from feature_learning.kan import KANDecoder, KANEncoder
from feature_learning.pca import PCAEncoder
from ml_approach.ae import AETrainer
from ml_approach.cl import CLTrainer
from ml_approach.dc import DCTrainer
from modality.raw_iq import raw_iq

METHODS = [("pca", "none"), ("kan", "ae"), ("kan", "dc"), ("kan", "cl"),
           ("cnn", "dc"), ("cnn", "cl")]                    # (feature learner, ML approach) pairs so far
METRICS = ["roc_auc", "f1", "precision", "recall", "accuracy", "ami", "nmi"]


def score_fold(cfg, z_train, z_test, fold, meta):
    """Detector + evaluation for one fold, given its features. The truth (meta) is used only here."""
    detector = KMeansDetector(n_clusters=cfg["n_clusters"], seed=cfg["seed"]).fit(z_train)
    scores, clusters = detector.score(z_test)
    test_tx = meta["tx"].iloc[fold["test"]].to_numpy()
    return evaluate(scores, clusters, test_tx == fold["unknown"], test_tx, cfg["threshold"])


def make_encoder(cfg, size):
    """The feature learner that gets trained. This is the only part that differs between KAN and CNN."""
    if cfg["feature_learner"] == "kan":
        return KANEncoder(in_features=size, n_features=cfg["n_features"],
                          grid_size=cfg["grid_size"], spline_order=cfg["spline_order"])
    if cfg["feature_learner"] == "cnn":
        return CNNEncoder(n_features=cfg["n_features"])


def make_trainer(cfg, encoder, size):
    """The ML approach that trains the encoder. This is the only part that differs between AE, DC and CL."""
    if cfg["ml_approach"] == "ae":
        decoder = KANDecoder(n_features=cfg["n_features"], out_features=size,
                             grid_size=cfg["grid_size"], spline_order=cfg["decoder_spline_order"])
        return AETrainer(encoder, decoder, lr=cfg["lr"], batch_size=cfg["batch_size"],
                         noise_std=cfg["noise_std"], seed=cfg["seed"], device=cfg["device"])
    if cfg["ml_approach"] == "dc":
        # No head (as the old code): the 20 features are the scores of the groups, so there must be 20 groups.
        assert cfg["n_groups"] == cfg["n_features"], "DC without a head needs n_groups == n_features"
        return DCTrainer(encoder, n_groups=cfg["n_groups"], update_every=cfg["update_every"], lr=cfg["lr"],
                         batch_size=cfg["batch_size"], seed=cfg["seed"], device=cfg["device"])
    if cfg["ml_approach"] == "cl":
        return CLTrainer(encoder, n_features=cfg["n_features"], lr=cfg["lr"], batch_size=cfg["batch_size"],
                         temperature=cfg["temperature"], aug_std=cfg["aug_std"], seed=cfg["seed"],
                         device=cfg["device"])



def run(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    method = (cfg["feature_learner"], cfg["ml_approach"])
    if method not in METHODS:
        raise ValueError(f"Unknown feature learner + ML approach: {method}")

    out_dir = Path("results") / f"{date.today():%Y-%m-%d}_{cfg['name']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, out_dir / "config.yaml")          # keep the exact settings next to the results

    # Data, split and modality: done once, the same for every fold.
    signals, meta = load_manysig(equalized=cfg["equalized"], receivers=cfg["receivers"])
    x = raw_iq(signals, norm=cfg["norm"])
    folds = make_folds(meta, cfg["split"])

    rows = []
    for k, fold in enumerate(folds):
        x_train, x_test = x[fold["train"]], x[fold["test"]]
        print(f"--- fold {k} (unknown {fold['unknown']})")

        if method == ("pca", "none"):
            # One exact fit, so one checkpoint (epoch 0).
            encoder = PCAEncoder(n_features=cfg["n_features"]).fit(x_train)
            metrics = score_fold(cfg, encoder.encode(x_train), encoder.encode(x_test), fold, meta)
            rows.append({"fold": k, "unknown": fold["unknown"], "epoch": 0, **metrics})
            print("  ".join(f"{m} {metrics[m]:.3f}" for m in METRICS))

        else:
            # KAN or CNN, trained on x_train only (AE, DC or CL: only the trainer differs). Every eval_every
            # epochs, score the test set and write it down; those scores never feed back into training.
            torch.manual_seed(cfg["seed"])                    # same starting weights every run
            size = x_train[0].size                            # 2 x 256 = 512 numbers per signal
            encoder = make_encoder(cfg, size)
            trainer = make_trainer(cfg, encoder, size)

            for epoch in range(1, cfg["n_epochs"] + 1):
                loss = trainer.train_epoch(x_train)
                if epoch % cfg["eval_every"] == 0:
                    metrics = score_fold(cfg, encoder.encode(x_train), encoder.encode(x_test), fold, meta)
                    rows.append({"fold": k, "unknown": fold["unknown"], "epoch": epoch, "loss": loss, **metrics})
                    print(f"epoch {epoch:3d}  loss {loss:.5f}  roc_auc {metrics['roc_auc']:.3f}  "
                          f"f1 {metrics['f1']:.3f}  ami {metrics['ami']:.3f}")

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "folds.csv", index=False)

    # "final": the last epoch of every fold (honest). "best_checkpoint": per fold, the checkpoint with the best
    # TEST roc_auc (old style, a leak; kept only to compare with the old table). For PCA they are the same.
    final = table[table["epoch"] == table["epoch"].max()]
    best = table.loc[table.groupby("fold")["roc_auc"].idxmax()]
    summary = pd.concat({"final": final[METRICS].agg(["mean", "std"]).T,
                         "best_checkpoint": best[METRICS].agg(["mean", "std"]).T}, axis=1)
    summary.to_csv(out_dir / "summary.csv")
    print(summary.round(3))


if __name__ == "__main__":
    run(sys.argv[1])