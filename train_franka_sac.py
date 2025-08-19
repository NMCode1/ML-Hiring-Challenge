import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import gymnasium as gym
import sai_mujoco  # <-- THIS registers the Franka envs with Gym

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

ENV_ID = "FrankaIkGolfCourseEnv-v0"  # keep the -v0 suffix
SEED = 42
TOTAL_STEPS = 2_000_000


# Environment ID from SAI
ENV_ID = "FrankaIkGolfCourseEnv-v0"
SEED = 42
TOTAL_STEPS = 2_000_000  # Can increase later to 5M+ for higher performance

# Create environment function
def make_env():
    env = gym.make(ENV_ID)
    env = Monitor(env)
    return env

# Vectorized environment (single process for now)
env = DummyVecEnv([make_env])

# Policy architecture
policy_kwargs = dict(net_arch=[256, 256])

# Initialize SAC model
model = SAC(
    "MlpPolicy",
    env,
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

# Evaluation environment
eval_env = DummyVecEnv([make_env])

# Callbacks for evaluation and checkpoints
eval_cb = EvalCallback(
    eval_env,
    best_model_save_path="./models/franka_sac/",
    log_path="./logs/franka_sac/",
    eval_freq=10_000,
    n_eval_episodes=10,
    deterministic=False,
    render=False,
)
ckpt_cb = CheckpointCallback(
    save_freq=100_000,
    save_path="./models/franka_sac/",
    name_prefix="ckpt"
)

# Train the model
model.learn(total_timesteps=TOTAL_STEPS, callback=[eval_cb, ckpt_cb])

# Save the best model
best_path = "./models/franka_sac/best_model.zip"
model.save(best_path)
print(f"\n✅ Saved best model to: {os.path.abspath(best_path)}")
