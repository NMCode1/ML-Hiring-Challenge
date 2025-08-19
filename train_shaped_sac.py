# train_shaped_sac.py
import os
from sai_rl import SAIClient
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from wrappers import RescaleAction, RewardShapingWrapper

COMP_ID = "franka-ml-hiring"
TOTAL_STEPS = 2_000_000
N_ENVS_TRAIN = 4
N_ENVS_EVAL  = 2
EVAL_FREQ = 50_000

sai = SAIClient(comp_id=COMP_ID)

def make_env_train():
    env = sai.make_env()  # exact competition config
    # small action range for control (tune 0.12–0.25)
    minmax = 0.20
    env = RescaleAction(env, [-minmax]*env.action_space.shape[0], [minmax]*env.action_space.shape[0])
    env = RewardShapingWrapper(env)
    env = Monitor(env)
    return env

def make_env_eval():
    env = sai.make_env()  # eval WITHOUT shaping, to approximate comp reward
    minmax = 0.20
    env = RescaleAction(env, [-minmax]*env.action_space.shape[0], [minmax]*env.action_space.shape[0])
    env = Monitor(env)
    return env

train_env = DummyVecEnv([make_env_train for _ in range(N_ENVS_TRAIN)])
eval_env  = DummyVecEnv([make_env_eval  for _ in range(N_ENVS_EVAL)])

# probe action dim for target entropy
probe = sai.make_env()
action_dim = int(probe.action_space.shape[0])
probe.close()
target_entropy = -0.5 * action_dim  # higher exploration early

policy_kwargs = dict(net_arch=[400, 300])

model = SAC(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    buffer_size=1_000_000,
    batch_size=1024,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    target_update_interval=1,
    ent_coef="auto",
    target_entropy=target_entropy,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log="./logs/tb_shaped/",
    device="auto",
)

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path="./models/shaped_sac/",
    log_path="./logs/shaped_sac/",
    eval_freq=EVAL_FREQ,
    n_eval_episodes=10,
    deterministic=True,
)

ckpt_cb = CheckpointCallback(
    save_freq=100_000,
    save_path="./models/shaped_sac/",
    name_prefix="ckpt",
)

model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=[eval_cb, ckpt_cb],
    progress_bar=True,
)

# Save final as well
best_path = "./models/shaped_sac/best_model.zip"
final_path = "./models/shaped_sac/final_model.zip"
model.save(final_path)
print(f"\n✅ Saved best (exists if eval improved): {os.path.abspath(best_path)}")
print(f"✅ Saved final: {os.path.abspath(final_path)}")
