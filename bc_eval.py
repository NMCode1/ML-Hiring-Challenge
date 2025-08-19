# bc_eval.py
import os, json, time
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from sai_rl import SAIClient

SAVE_DIR = "models/bc"

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, out_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def main():
    with open(os.path.join(SAVE_DIR, "meta.json"), "r") as f:
        meta = json.load(f)
    obs_mean = np.array(meta["obs_mean"], dtype=np.float32)
    obs_std  = np.array(meta["obs_std"],  dtype=np.float32)
    obs_dim  = meta["obs_dim"]
    act_dim  = meta["act_dim"]
    hidden   = meta["hidden"]

    model = MLP(obs_dim, act_dim, hidden)
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model.pt"), map_location="cpu"))
    model.eval()

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create SAI env with rendering
    sai = SAIClient(comp_id="franka-ml-hiring")
    try:
        env = sai.make_env(render_mode="human")
    except TypeError:
        env = sai.make_env(env_kwargs={"render_mode": "human"})

    obs, info = env.reset(seed=0)
    done = False
    steps = 0
    rew_sum = 0.0

    def act_fn(o):
        x = (o - obs_mean) / (obs_std + 1e-8)
        x = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            a = model(x).cpu().numpy()[0]
        return np.clip(a, env.action_space.low, env.action_space.high)

    try:
        while True:
            a = act_fn(obs)
            obs, r, term, trunc, info = env.step(a)
            steps += 1
            rew_sum += float(r)
            try:
                env.render()
            except Exception:
                pass
            if term or trunc:
                print(f"Episode done. Steps={steps}, reward_sum≈{rew_sum:.3f}")
                steps = 0
                rew_sum = 0.0
                obs, info = env.reset()
            time.sleep(0.02)
    finally:
        env.close()

if __name__ == "__main__":
    main()
