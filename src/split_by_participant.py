from collections import defaultdict
import random
import numpy as np

def group_paths_by_pid(npz_paths):
    # Each file holds multiple pids; we’ll let the Dataset handle mixing.
    # Here we just return the list; true PID split is done after loading metadata if needed.
    return npz_paths

def split_indices_by_pid(dataset, val_pid_list):
    idx_train, idx_val = [], []
    for i in range(len(dataset)):
        _, _, pid, _ = dataset[i]
        (idx_val if pid in val_pid_list else idx_train).append(i)
    return idx_train, idx_val

def subset_from_indices(dataset, indices):
    from torch.utils.data import Subset
    return Subset(dataset, indices)

def leave_one_pid_out(npz_paths_all, held_out_pid):
    # Build a temporary dataset only to inspect metadata/pids
    tmp = FogNPZ(npz_paths_all, transform=None)
    idx_train, idx_val = split_indices_by_pid(tmp, [held_out_pid])
    train_subset = subset_from_indices(tmp, idx_train)
    val_subset   = subset_from_indices(tmp, idx_val)
    return train_subset, val_subset
