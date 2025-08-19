# train_ml_hiring_sac.py
import os
from sai_rl import SAIClient
import gymnasium as gym

# Optional: ensures Franka envs are registered in some setups
try:
    import sai_mujoco  # noqa: F401
except Exception:
    pass

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

COMP_ID = "franka-ml-hiring"  # <- ML Hiring Challenge
SEED = 42
TOTAL_STEPS = 2_000_000  # run longer later (5–10M) for better results

# 1) Connect to SAI challenge
sai = SAIClient(comp_id=COMP_ID)
env = sai.make_env()
# 2) Build train/eval envs from the challenge env
def make_env():
    env = sai.make_env()     # official ML Hiring Challenge env
    env = Monitor(env)       # track episode returns/lengths
    return env

train_env = DummyVecEnv([make_env])
eval_env = DummyVecEnv([make_env])

# 3) SAC model (strong baseline for continuous control)
policy_kwargs = dict(net_arch=[256, 256])
model = SAC(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    buffer_size=1_000_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    target_update_interval=1,
    ent_coef="auto",
    policy_kwargs=policy_kwargs,
    verbose=1,
    seed=SEED,
)

# 4) Callbacks: eval & checkpoints
eval_cb = EvalCallback(
    eval_env,
    best_model_save_path="./models/ml_hiring_sac/",
    log_path="./logs/ml_hiring_sac/",
    eval_freq=10_000,
    n_eval_episodes=10,
    deterministic=False,
)

ckpt_cb = CheckpointCallback(
    save_freq=100_000,
    save_path="./models/ml_hiring_sac/",
    name_prefix="ckpt",
)

# 5) Train
model.learn(total_timesteps=TOTAL_STEPS, callback=[eval_cb, ckpt_cb])

# 6) Save best model
best_path = "./models/ml_hiring_sac/best_model.zip"
model.save(best_path)
print(f"\n✅ Best model saved at: {os.path.abspath(best_path)}")

# 7) Submit directly to the ML Hiring Challenge
from sai_rl import SAIClient
from stable_baselines3 import SAC

# Connect to the ML Hiring Challenge
sai = SAIClient(comp_id="franka-ml-hiring")

# Load your trained SB3 model
model = SAC.load("./models/ml_hiring_sac/best_model.zip")

# Submit it (note: method is submit_model, and model_type must match)
result = sai.submit_model(
    name="Nick SAC v1",
    model=model,
    model_type="stable_baselines3"
)

print("Submitted. Result:", result)


print("🚀 Submitted to ML Hiring Challenge (franka-ml-hiring).")
