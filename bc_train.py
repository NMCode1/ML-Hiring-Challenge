# bc_train.py  — Behavior Cloning with stability tweaks
import os, json, glob, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =========================
# Config
# =========================
SAVE_DIR     = "models/bc"
DEMOS_GLOB   = "demos/*.npz"
SEED         = 123
BATCH_SIZE   = 2048        # try 1024 if you hit OOM
LR           = 3e-4
EPOCHS       = 200       # ceiling; early stopping halts sooner
VAL_SPLIT    = 0.10
WEIGHT_DECAY = 1e-5
HIDDEN       = [512, 512]

# Stability knobs
ACTION_WEIGHTS = [1, 1, 1, 1, 1, 1, 0.3]  # down-weight gripper (assumed last dim)
SMOOTH_LAMBDA  = 1e-3                     # penalize action deltas (stability)
PATIENCE       = 10                     # early stopping patience epochs
MIN_LR         = 1e-5                     # floor for LR scheduler

# =========================
# Repro & paths
# =========================
os.makedirs(SAVE_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# =========================
# Data
# =========================
class NpzDemoDataset(Dataset):
    def __init__(self, obs, act, mean=None, std=None):
        self.obs = obs.astype(np.float32)
        self.act = act.astype(np.float32)
        if mean is None:
            self.mean = self.obs.mean(axis=0)
            self.std  = self.obs.std(axis=0) + 1e-8
        else:
            self.mean, self.std = mean, std

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, i):
        x = (self.obs[i] - self.mean) / self.std
        y = self.act[i]
        return torch.from_numpy(x), torch.from_numpy(y)

def load_all_demos(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No demo files found at {pattern}")
    obs_list, act_list = [], []
    total_steps = 0
    for f in files:
        d = np.load(f)
        obs, act = d["obs"], d["act"]
        assert obs.shape[0] == act.shape[0], f"Length mismatch in {f}"
        obs_list.append(obs)
        act_list.append(act)
        total_steps += obs.shape[0]
    obs = np.concatenate(obs_list, axis=0)
    act = np.concatenate(act_list, axis=0)
    return obs, act, files, total_steps

# =========================
# Model
# =========================
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        # Tanh → actions in [-1, 1]
        layers += [nn.Linear(last, out_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def weighted_mse(pred, target, w=None):
    # pred/target: (B, act_dim); w: (act_dim,)
    if w is None:
        return F.mse_loss(pred, target)
    return ((w * (pred - target) ** 2).mean())

# =========================
# Train
# =========================
def main():
    # Load data
    obs, act, files, total = load_all_demos(DEMOS_GLOB)
    print(f"Loaded {len(files)} files, total steps = {total:,}")
    n, obs_dim = obs.shape
    act_dim = act.shape[1]
    print(f"Obs dim = {obs_dim}, Act dim = {act_dim}")

    # Shuffle & split by index (simple; later we can split by file to avoid leakage)
    idx = np.arange(n)
    np.random.shuffle(idx)
    val_n = int(n * VAL_SPLIT)
    val_idx, train_idx = idx[:val_n], idx[val_n:]

    train_ds = NpzDemoDataset(obs[train_idx], act[train_idx])  # learns mean/std
    val_ds   = NpzDemoDataset(obs[val_idx],   act[val_idx],
                              mean=train_ds.mean, std=train_ds.std)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # Model/opt
    model = MLP(in_dim=obs_dim, out_dim=act_dim, hidden=HIDDEN)
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Device:", device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, factor=0.5, patience=2, min_lr=MIN_LR
)


    # Prep action weights
    aw = None
    if ACTION_WEIGHTS is not None:
        assert len(ACTION_WEIGHTS) == act_dim, "ACTION_WEIGHTS length must match act_dim"
        aw = torch.tensor(ACTION_WEIGHTS, dtype=torch.float32, device=device)

    # Early stopping
    best_val = float("inf")
    bad = 0

    for epoch in range(1, EPOCHS + 1):
        # ---- train ----
        model.train()
        total_loss = 0.0
        total_bc   = 0.0
        total_sm   = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)

            # Imitation loss (weighted per action dim)
            bc_loss = weighted_mse(pred, yb, aw)

            # Smoothness penalty (cheap within-batch proxy)
            if yb.size(0) > 1:
                delta = yb[1:] - yb[:-1]
                smooth_loss = (delta ** 2).mean()
            else:
                smooth_loss = torch.zeros((), device=device)

            loss = bc_loss + SMOOTH_LAMBDA * smooth_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            total_loss += loss.item()   * xb.size(0)
            total_bc   += bc_loss.item()* xb.size(0)
            total_sm   += smooth_loss.item()* xb.size(0)

        train_loss = total_loss / len(train_ds)
        train_bc   = total_bc   / len(train_ds)
        train_sm   = total_sm   / len(train_ds)

        # ---- val ----
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                vloss += F.mse_loss(pred, yb).item() * xb.size(0)
        val_loss = vloss / max(len(val_ds), 1)

        # LR schedule step on val metric
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:03d} | "
            f"train MSE {train_loss:.6f} (bc {train_bc:.6f} + sm {train_sm:.6f}) | "
            f"val MSE {val_loss:.6f} | "
            f"lr {opt.param_groups[0]['lr']:.2e}"
        )           


        # ---- checkpoint / early stop ----
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            bad = 0
            # save best weights + normalization stats
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model.pt"))
            meta = {
                "obs_mean": train_ds.mean.tolist(),
                "obs_std":  train_ds.std.tolist(),
                "obs_dim":  obs_dim,
                "act_dim":  act_dim,
                "hidden":   HIDDEN,
                "val_mse":  float(best_val),
            }
            with open(os.path.join(SAVE_DIR, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            print("  ↳ saved best to models/bc/")
        else:
            bad += 1
            if bad >= PATIENCE:
                print("Early stopping (no val improvement).")
                break

    print("Done.")

if __name__ == "__main__":
    main()
