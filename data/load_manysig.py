"""Load the WiSig ManySig subset: pick signals by transmitter, receiver, day and version."""


import pickle
from pathlib import Path

import numpy as np
import pandas as pd


PKL = Path(__file__).parent / "raw" / "wisig_manysig" / "ManySig.pkl"


def load_manysig(equalized, receivers, transmitters=None, days=None, path=PKL):
    """Return (signals, meta) for the chosen part of ManySig.

    equalized (True/False) and receivers (list of names, or None = all) have no default,
    so every experiment must state them. transmitters and days: None = all.

    signals: float32 array (N, 256, 2) with I and Q, values as in the file (not normalized).
    meta:    table with one row per signal: tx, rx, day, idx (position within its box, 0-999).
             Row i of meta describes signals[i].
    """

# Open file
    with open(path, "rb") as f:
        ds = pickle.load(f)


    if receivers is None:                                   # if it wasn't said which RX was
        receivers = ds["rx_list"]                           # take all 12
    if transmitters is None:
        transmitters = ds["tx_list"]
    if days is None:                                        # if days were none
        days = ds["capture_date_list"]                      # take all 4
    e = ds["equalized_list"].index(1 if equalized else 0)   # find where equalized version is

    blocks, rows = [], []
    for tx in transmitters:                                 # go through the boxes, one by one
        t = ds["tx_list"].index(tx)                         # 6 transmitters
        for rx in receivers:                                # 1 receiver
            r = ds["rx_list"].index(rx)
            for day in days:                                # days
                d = ds["capture_date_list"].index(day)
                block = ds["data"][t][r][d][e]              # one box, 1000 signals, (1000, 256, 2)
                blocks.append(block.astype(np.float32))     # put the signals on the pile
                rows += [(tx, rx, day, i) for i in range(len(block))]       # write 1000 labels
                                                            # 6x1x4=24 boxes - for each box, it does two things at the same time
                                                            #   puts its 1000 signals on the pile blocks
                                                            #   writes 1000 rows saying where they came from
    # stack, label, check, return
    signals = np.concatenate(blocks)                        # 24 boxes -> (24 000, 256, 2) ---n fits only call V0/V1
    meta = pd.DataFrame(rows, columns=["tx", "rx", "day", "idx"])   # 24 000 rows
    assert len(signals) == len(meta)                        # stop if they don't match
    return signals, meta


if __name__ == "__main__":
    # Quick check: the old setup (equalized, receiver 1-1, all tx, all days) should give 24,000 signals.
    signals, meta = load_manysig(equalized=True, receivers=["1-1"])
    print(signals.shape, signals.dtype)
    print(meta["tx"].value_counts())


# META LOOKS LIKE THIS
#           tx          rx          day             idx
#       0   14-10       1-1         2021_03_01      0
#       1   14-10       1-1         2021_03_01      1
#       ......
#     999   14-10       1-1         2021_03_01      999
#    1000   14-10       1-1         2021_03_08      0
#       ......
#   23999    8-20       1-1         2021_03_23      999












