"""Raw I/Q modality: the signal itself, as the model input. Never sees labels."""
### NORMALIZATION - needed because signal's loudness depends mostly on distance to the rx.
#   if signals are kept loud - the model might learn /this device is far away/ instead of /this is device X/
#   dividing by 'loudness' keeps the shape, which is where fingerprint is 
#   since this is modeling choice, it's separate, and not just in the loader

import numpy as np

def raw_iq(signals, norm):
    # Turn loaded signals into model input
    # signals: float32 array (N, 256,2) from a loader (I and Q last)
    # norm: "max_abs"=divide each signal by its own largest |value| over I and Q (as the old code)
    #       "none"=leave the values as they are
    # returns: float32 array (N, 2, 256): I and Q first, then time (the order of old code and pythorch use)
    # N - position, value here, how many signals; 2 - I and Q, two parts of each signal; 
    # 256 - time, 256 moments, total 10.24microsec, every moment 0.04microsec, because of 25 MS/s

    if norm not in ("max_abs", "none"):
        raise ValueError(f"Unknown norm: {norm}")

    x = signals.transpose(0, 2, 1)                          # (N, 256, 2) -> (N, 2, 256)
    if norm == "max_abs":
        peak = np.abs(x).max(axis=(1, 2), keepdims=True)    # one number per signal
        assert (peak > 0).all()                             # an all-zero signal would divide by 0
        x = x / peak
    return x.astype(np.float32)


if __name__ == "__main__":
    # Quick check on v0/v1 data: shape, every peak = 1, and the same result as the old formula.
    from data.load_manysig import load_manysig

    signals, meta = load_manysig(equalized=True, receivers=["1-1"])
    x = raw_iq(signals, norm="max_abs")
    print(x.shape, x.dtype)

    peaks = np.abs(x).max(axis=(1, 2))
    print("peak per signal: min", peaks.min(), "max", peaks.max())

    old = signals[0].T / np.max(np.abs(signals[0].T))      # the old code's formula, on signal 0
    print("same as old code:", np.allclose(x[0], old))