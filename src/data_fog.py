import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class FogNPZ(Dataset):
    """
    Returns:
        x: FloatTensor [C, T]  (6, 120)
        y: FloatTensor [1, T]  (1, 120)  (label per frame; dataset provides a window label)
        pid: str               participant id, e.g., 'PD012'
        meta: str              raw metadata string
    """
    def __init__(self, npz_paths, transform=None, label_as_framewise=True):
        xs, ys, metas = [], [], []
        for p in npz_paths:
            with np.load(p) as d:
                xs.append(d['xTensor'])       # (N,120,6)
                ys.append(d['yTensor'])       # (N,)
                metas.append(d['Metadata'])   # (N,)
        self.x = np.concatenate(xs, axis=0).astype(np.float32)   # (N,120,6) -> float32
        self.y = np.concatenate(ys, axis=0).astype(np.int64)     # (N,)
        self.meta = np.concatenate(metas, axis=0).astype(str)    # (N,)

        self.transform = transform
        self.label_as_framewise = label_as_framewise

    @staticmethod
    def pid_from_meta(meta_str: str) -> str:
        # Expect things like "... PD0XX ..." — keep the whole token if present
        m = re.search(r'PD\d+', meta_str.upper())
        return m.group(0) if m else "UNKNOWN"

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        # To NCL
        x = torch.from_numpy(self.x[i]).transpose(0,1)  # (120,6)->(6,120)
        y_scalar = int(self.y[i])
        if self.label_as_framewise:
            y = torch.full((1, x.shape[-1]), float(y_scalar), dtype=torch.float32)  # (1,120)
        else:
            y = torch.tensor([y_scalar], dtype=torch.float32)  # (1,)
        meta = self.meta[i]
        pid = self.pid_from_meta(meta)

        if self.transform:
            x = self.transform(x)
        return x, y, pid, meta


def make_loaders(npz_paths_train, npz_paths_val, batch_size=128, num_workers=0):
    ds_train = FogNPZ(npz_paths_train)
    ds_val   = FogNPZ(npz_paths_val)

    # Compute per-channel mean/std from training set only
    X = torch.from_numpy(np.concatenate([np.load(p)['xTensor'] for p in npz_paths_train], axis=0)).float()  # (N,120,6)
    # To NCL then merge batch/time to compute stats per channel
    X = X.transpose(1,2).contiguous()  # (N,6,120)
    mean = X.mean(dim=(0,2))           # (6,)
    std  = X.std(dim=(0,2)).clamp_min(1e-6)

    def norm(x):  # x: (6,T)
        return (x - mean.view(-1,1)) / std.view(-1,1)

    ds_train.transform = norm
    ds_val.transform   = norm

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    return train_loader, val_loader, mean, std
