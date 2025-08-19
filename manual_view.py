# manual_view.py
import gymnasium as gym
from sai_rl import SAIClient
import os

# On mac, this makes mujoco use a GL backend that works
os.environ.setdefault("MUJOCO_GL", "glfw")

sai = SAIClient(comp_id="franka-ml-hiring")

# 🔑 set render_mode at creation time
# Try both ways (some SAI versions use env_kwargs)
try:
    env = sai.make_env(render_mode="human")
except TypeError:
    env = sai.make_env(env_kwargs={"render_mode": "human"})

# Minimal loop just to see the window
obs, info = env.reset(seed=0)
for t in range(600):
    # random actions just to see motion
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
    # Most gymnasium envs with render_mode="human" auto-render each step

env.close()
print("Done.")
