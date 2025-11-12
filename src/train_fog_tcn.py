import torch
from torch.utils.data import DataLoader
from pytorch_tcn import TCN
from data_fog import FogNPZ, make_loaders
import glob, os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- choose which dataset files you want ----
# Example: all “AllActivities” for training
train_files = sorted(glob.glob('data/PD*_AllActivities_FOG.npz'))
val_files   = sorted(glob.glob('data/PD*_WalkingTurning_FOG.npz'))  # or another subject split

train_loader, val_loader, mean, std = make_loaders(train_files, val_files, batch_size=128, num_workers=4)

# ---- TCN configuration for per-frame logits ----
model = TCN(
    num_inputs=6,
    num_channels=[32, 32, 64, 64],
    kernel_size=5,
    dilations=[1, 2, 4, 8],          # RF = 1 + (5-1)*(1+2+4+8) = 1 + 4*15 = 61 samples ≈ 1.02 s @ 60 Hz
    dropout=0.1,
    causal=True,
    use_norm='weight_norm',
    activation='relu',
    use_skip_connections=True,
    input_shape='NCL',
    output_projection=1,
    output_activation=None,
).to(device)

# ---- Class imbalance handling ----
# Compute positive fraction across train set quickly
pos, total = 0, 0
with torch.no_grad():
    for _, y, _, _ in train_loader:
        pos += y.sum().item()
        total += y.numel()
pos_frac = max(pos / total, 1e-6)
neg_frac = 1.0 - pos_frac
pos_weight = torch.tensor([neg_frac / pos_frac], device=device)   # ~ (#neg / #pos)

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

def f1_at_threshold(logits, targets, thr=0.5):
    # logits: [B,1,T], targets: [B,1,T]
    probs = torch.sigmoid(logits)
    preds = (probs >= thr).float()
    tp = (preds*targets).sum()
    fp = (preds*(1-targets)).sum()
    fn = ((1-preds)*targets).sum()
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1 = 2*precision*recall / (precision + recall + 1e-8)
    return f1.item(), precision.item(), recall.item()

best_val = float('inf')
for epoch in range(30):
    model.train()
    running = 0.0
    for x, y, _, _ in train_loader:
        x, y = x.to(device), y.to(device)       # x: [B,6,120], y: [B,1,120]
        logits = model(x)                       # [B,1,120]
        loss = criterion(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running += loss.item() * x.size(0)
    train_loss = running / len(train_loader.dataset)

    # quick validation
    model.eval()
    vloss, vf1, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y, _, _ in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            vloss += criterion(logits, y).item() * x.size(0)
            f1, _, _ = f1_at_threshold(logits, y, thr=0.5)
            vf1 += f1 * x.size(0)
            n += x.size(0)
    vloss /= len(val_loader.dataset)
    vf1 /= max(n,1)
    print(f"epoch {epoch:02d}  train_bce={train_loss:.4f}  val_bce={vloss:.4f}  val_f1@0.5={vf1:.3f}")

    if vloss < best_val:
        best_val = vloss
        torch.save({'state_dict': model.state_dict(), 'mean': mean, 'std': std}, 'tcn_fog_best.pt')
