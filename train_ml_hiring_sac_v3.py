# train_ml_hiring_sac_v3.py (Explorer)
# Goal: keep exploration alive to discover the full chain (grasp → hit → hole),
# while maintaining stability and good throughput.

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
SEED = 123                 # run more seeds later: 42, 2025, etc.
TOTAL_STEPS = 5_000_000    # push to 10M overnight if learning keeps improving
N_ENVS_TRAIN = 6           # higher parallelism for more FPS
N_ENVS_EVAL  = 2           # reduce eval overhead
EVAL_FREQ = 50_000         # evaluate less often to save time

# Connect to the official challenge
sai = SAIClient(comp_id=COMP_ID)

# Probe action dim to set a higher target entropy (keeps exploration from collapsing)
probe_env = sai.make_env()
assert hasattr(probe_env.action_space, "shape"), "Action space must be continuous"
action_dim = int(probe_env.action_space.shape[0])
probe_env.close()

# Encourage more exploration than SAC default (-action_dim)
TARGET_ENTROPY = -0.5 * action_dim   # less negative => higher entropy target


def make_env():
    env = sai.make_env()
    env = Monitor(env)
    return env


# Parallelized envs
train_env = DummyVecEnv([make_env for _ in range(N_ENVS_TRAIN)])
eval_env  = DummyVecEnv([make_env for _ in range(N_ENVS_EVAL)])


# Simple linear LR schedule (decays to 0 over training)

def linear_schedule(initial_value: float):
    def f(progress_remaining: float) -> float:
        # progress_remaining goes from 1.0 -> 0.0
        return initial_value * progress_remaining
    return f


# Larger policy; batch up for stability
policy_kwargs = dict(net_arch=[512, 512])

model = SAC(
    "MlpPolicy",
    train_env,
    learning_rate=linear_schedule(3e-4),
    buffer_size=1_000_000,
    batch_size=512,
    learning_starts=20_000,      # give replay buffer time before updates
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,            # keep stable; 1:1 updates to samples
    target_update_interval=1,    # default/steady target updates
    ent_coef="auto",            # automatic entropy tuning
    target_entropy=TARGET_ENTROPY,
    policy_kwargs=policy_kwargs,
    verbose=1,
    seed=SEED,
    device="auto",
    tensorboard_log="./logs/tb_v3/",  # optional; run: tensorboard --logdir ./logs
)

# Evaluate regularly; save the best checkpoint automatically (deterministic eval)
eval_cb = EvalCallback(
    eval_env,
    best_model_save_path="./models/ml_hiring_sac_v3/",
    log_path="./logs/ml_hiring_sac_v3/",
    eval_freq=EVAL_FREQ,
    n_eval_episodes=10,
    deterministic=True,
)

ckpt_cb = CheckpointCallback(
    save_freq=100_000,
    save_path="./models/ml_hiring_sac_v3/",
    name_prefix="ckpt",
)

# Train with progress bar for ETA
model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=[eval_cb, ckpt_cb],
    progress_bar=True,
)

# Save and submit best model
best_path = "./models/ml_hiring_sac_v3/best_model.zip"
model.save(best_path)
print(f"\n✅ Best model saved at: {os.path.abspath(best_path)}")
