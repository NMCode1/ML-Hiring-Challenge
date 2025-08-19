# bc_model.py
import json, os
import numpy as np
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, out_dim), nn.Tanh()]  # SAC-style actions in [-1,1]
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: (B, obs_dim) float32 tensor
        return self.net(x)

class BCPolicy(nn.Module):
    """
    Self-contained, SAI-compliant policy:
      - Normalizes observations using saved mean/std
      - Runs MLP
      - Clips to [-1, 1]
    """
    def __init__(self, meta_path: str, weights_path: str):
        super().__init__()
        with open(meta_path, "r") as f:
            meta = json.load(f)
        self.obs_mean = torch.tensor(meta["obs_mean"], dtype=torch.float32)
        self.obs_std  = torch.tensor(meta["obs_std"],  dtype=torch.float32)
        obs_dim = meta["obs_dim"]
        act_dim = meta["act_dim"]
        hidden  = meta["hidden"]

        self.model = MLP(obs_dim, act_dim, hidden)
        state = torch.load(weights_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, obs_dim) float32 tensor
        returns: (B, act_dim) float32 tensor in [-1, 1]
        """
        x = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        a = self.model(x)
        return torch.clamp(a, -1.0, 1.0)
