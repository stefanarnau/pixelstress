from pathlib import Path
import numpy as np
import pandas as pd

# Path vars
PATH_IN = Path("/mnt/data_dump/pixelstress/3_sequence_tfr/")

# Load data
X_seq = np.load(PATH_IN / "X_seq.npy")              # sequences x channels x freqs x times
freqs = np.load(PATH_IN / "freqs.npy")
times = np.load(PATH_IN / "times.npy")
channel_labels = np.load(PATH_IN / "channel_labels.npy", allow_pickle=True)

seq_meta = pd.read_csv(PATH_IN / "seq_meta.csv")

# restore categorical variables
seq_meta["group"] = pd.Categorical(
    seq_meta["group"],
    categories=["control", "experimental"]
)

seq_meta["half"] = pd.Categorical(
    seq_meta["half"],
    categories=["first", "second"]
)

# ensure numeric types
seq_meta["f"] = seq_meta["f"].astype(float)
seq_meta["f_quad"] = seq_meta["f_quad"].astype(float)
seq_meta["mean_trial_difficulty"] = seq_meta["mean_trial_difficulty"].astype(float)
seq_meta["subject"] = seq_meta["subject"].astype(int)