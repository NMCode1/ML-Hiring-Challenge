# train_ml_hiring_sac_v2.py (Finisher / Stable SAC)
# Goal: stabilize critic, refine timing (grasp→hit→hole fast), no auto-submit.

import os
from sai_rl import SAIClient
import gymnasium as gym

# Some setups need this import to register Franka envs
try:
    import sai_mujoco  # noqa: F401
except Exception:
    pass

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

COMP_ID = "franka-ml-hiring"
SEED = 42
TOTAL_STEPS = 5_000_000   # bump to 10M if improving
N_ENVS_TRAIN = 6          # more parallelism
N_ENVS_EVAL  = 2
EVAL_FREQ = 50_000        # reduce eval overhead

# Connect to the official ML Hiring Challenge
sai = SAIClient(comp_id=COMP_ID)

def make_env():
    env = sai.make_env()     # official challenge env
    env = Monitor(env)
    return env

# Parallelized vector envs
train_env = DummyVecEnv([make_env for _ in range(N_ENVS_TRAIN)])
eval_env  = DummyVecEnv([make_env for _ in range(N_ENVS_EVAL)])

# Linear LR schedule for stability

def linear_schedule(initial_value: float):
    def f(progress_remaining: float) -> float:
        return initial_value * progress_remaining
    return f

# Stronger-but-stable network and batch size
policy_kwargs = dict(net_arch=[400, 300])

model = SAC(
    "MlpPolicy",
    train_env,
    learning_rate=linear_schedule(1e-4),  # lower LR for critic stability
    buffer_size=1_000_000,
    batch_size=1024,                      # larger batch for steadier updates
    learning_starts=10_000,               # brief warmup before learning
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    target_update_interval=1,
    ent_coef="auto",                     # default target entropy (-|A|)
    policy_kwargs=policy_kwargs,
    verbose=1,
    seed=SEED,
    device="auto",                       # use MPS/GPU if available
    tensorboard_log="./logs/tb_v2/",     # optional: tensorboard --logdir ./logs
)

# Evaluate regularly; save the best checkpoint automatically (deterministic eval)
eval_cb = EvalCallback(
    eval_env,
    best_model_save_path="./models/ml_hiring_sac_v2/",
    log_path="./logs/ml_hiring_sac_v2/",
    eval_freq=EVAL_FREQ,
    n_eval_episodes=10,
    deterministic=True,
)

ckpt_cb = CheckpointCallback(
    save_freq=100_000,
    save_path="./models/ml_hiring_sac_v2/",
    name_prefix="ckpt",
)

# Train with progress bar for ETA
model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=[eval_cb, ckpt_cb],
    progress_bar=True,
)

# Save best model only (manual submit later)
best_path = "./models/ml_hiring_sac_v2/best_model.zip"
model.save(best_path)
print(f"\n✅ Best model saved at: {os.path.abspath(best_path)}")
