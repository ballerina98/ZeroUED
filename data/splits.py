"""Folds and train/test splits, built from the meta table only (never from the signals)."""


import numpy as np

def make_folds(meta, split):
    #### FOLD
    # 6 TX, 6 rounds of folding, in each round, different transmitter plays the 'new device'
    # Fold          Unknown (hidden during training)        Known (used for training)
    #   0           14-10                                   14-7, 20-15, 20-19, 6-15, 8-20
    #   1           14-7                                    14-10, 20-15, 20-19, 6-15, 8-20
    #   2           20-15                                   14-10, 14-7, 20-19, 6-15, 8-20
    #   3           20-19                                   14-10, 14-7, 20-15, 6-15, 8-20
    #   4           6-15                                    14-10, 14-7, 20-15, 20-19, 8-20
    #   5           8-20                                    14-10, 14-7, 20-15, 20-19, 6-15
    # the need for folding - if we hide only 6-15, it might give good results
    # but hiding 14-7, which may sound lot of like 14-10, would be hard to spot
    # so do for each fold, report score, and then average, with range from x to y
    # so each fold has dictionary - the name of unknown devices, and train/test row numbers - they point to signals

    #### SPLIT
    # 6 devices, 1000 signals per device * 4 days; 5 devices used in training, 1 is uknown
    # Because it's zero shot, that new device cannot have any signal - it was not seen in the training

    # old = V0 - as the old cold ran.   Train: the 5 known tx, all signals; 5*4 000=20 000
    #                                   Test: all 6tx, all signals; 6*4 000=24 000
    # new = V1 -                        Train: the 5 known tx, first 80% of every box (16 000)
    #                                   Test: all 6tx, 5*800 (known), + 1*800 from that one unknown - 
    #                                         Test logic follows from the paper, just with leak edited


    # [v0 + v1] Stop on a split name that doesn't exit, instead of guessing
    if split not in ("v0", "v1"):
        raise ValueError(f"Unknown split method: {split}")


    # [V1]- Mark the first 80% of every box. Same in every fold, so done once, before the loop
    #       Shuffle could put 2 consecutive signals one in train one in test, and those are most likely to look alike
    if split == "v1":
        box_size = meta.groupby(["tx", "rx", "day"])["idx"].transform("size")
        early = (meta["idx"]<0.8*box_size).to_numpy()

    folds = []

    # [v0 + v1]-    One fold per tx, in each fold, that tx is the unknown one
    for unknown in meta["tx"].unique():
        known = (meta["tx"] != unknown).to_numpy()

        if split == "v0":
            # [v0]
            train = known
            test = np.ones(len(meta), dtype=bool)
        elif split == "v1":
            train = known & early
            test = ~early

        train_rows = np.flatnonzero(train)
        test_rows = np.flatnonzero(test)

        # [v0 + v1] The unknown transmitter is never in training.
        assert unknown not in set(meta["tx"].iloc[train_rows])
        # [v1] No signal in both train and test. (v0 would fail this on purpose: that's the old leak.)
        if split == "v1":
            assert not np.intersect1d(train_rows, test_rows).size

        # [v0 + v1]
        folds.append({"unknown": unknown, "train": train_rows, "test": test_rows})
    return folds

if __name__ == "__main__":
    # Quick check on the data of v0 and v1 (same loader call): receiver 1-1, equalized.
    from data.load_manysig import load_manysig

    signals, meta = load_manysig(equalized=True, receivers=["1-1"])
    for split in ["v0", "v1"]:
        print(f"---split: {split}")
        for fold in make_folds(meta,split):
            unknown_in_test = (meta["tx"].iloc[fold["test"]]==fold["unknown"]).sum()
            print(f"unknown {fold['unknown']:6s} train {len(fold['train'])} test {len(fold['test'])}  "
                  f"(unknown in test: {unknown_in_test})")



