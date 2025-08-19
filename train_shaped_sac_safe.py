# train_shaped_sac_safe.py
import os, signal, sys
from sai_rl import SAIClient
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from wrappers import RescaleAction, RewardShapingWrapper

# ----- SETTINGS (tune these later) -----
COMP_ID = "franka-ml-hiring"
TOTAL_STEPS = 300_000        # small first run to verify saves
N_ENVS_TRAIN = 2             # keep light to confirm it works
N_ENVS_EVAL  = 1
EVAL_FREQ = 10_000           # save best frequently
CKPT_FREQ = 20_000           # save periodic checkpoints frequently
ACTION_RANGE = 0.20
OUT_DIR = "./models/shaped_sac_safe"
LOG_DIR = "./logs/shaped_sac_safe"

# Create dirs up front so saves never fail
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

sai = SAIClient(comp_id=COMP_ID)

def make_env_train():
    env = sai.make_env()
    env = RescaleAction(env, [-ACTION_RANGE]*env.action_space.shape[0],
                             [ ACTION_RANGE]*env.action_space.shape[0])
    env = RewardShapingWrapper(env)
    env = Monitor(env)
    return env

def make_env_eval():
    env = sai.make_env()  # eval without shaping to approximate leaderboard
    env = RescaleAction(env, [-ACTION_RANGE]*env.action_space.shape[0],
                             [ ACTION_RANGE]*env.action_space.shape[0])
    env = Monitor(env)
    return env

train_env = DummyVecEnv([make_env_train for _ in range(N_ENVS_TRAIN)])
eval_env  = DummyVecEnv([make_env_eval  for _ in range(N_ENVS_EVAL)])

# Probe action dim for exploration target
probe = sai.make_env()
action_dim = int(probe.action_space.shape[0])
probe.close()
target_entropy = -0.5 * action_dim

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
    tensorboard_log=LOG_DIR,
    device="auto",
)

# Callbacks: best model + periodic checkpoints
eval_cb = EvalCallback(
    eval_env,
    best_model_save_path=OUT_DIR,
    log_path=LOG_DIR,
    eval_freq=EVAL_FREQ,
    n_eval_episodes=10,
    deterministic=True,
)

ckpt_cb = CheckpointCallback(
    save_freq=CKPT_FREQ,
    save_path=OUT_DIR,
    name_prefix="ckpt",
    save_replay_buffer=False,
    save_vecnormalize=False,
)

# Save an immediate “startup” snapshot so you always have a file
startup_path = os.path.join(OUT_DIR, "startup_model.zip")
model.save(startup_path)
print(f"✅ Wrote startup snapshot: {os.path.abspath(startup_path)}")

# Graceful interrupt: save a snapshot on Ctrl-C
def handle_sigint(sig, frame):
    snap = os.path.join(OUT_DIR, "interrupt_snapshot.zip")
    model.save(snap)
    print(f"\n🧷 Saved interrupt snapshot at {os.path.abspath(snap)}")
    sys.exit(0)
signal.signal(signal.SIGINT, handle_sigint)

# Train (short run to prove saves happen)
model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=[eval_cb, ckpt_cb],
    progress_bar=True,
)

# Always write a final model
final_path = os.path.join(OUT_DIR, "final_model.zip")
model.save(final_path)
print(f"\n✅ Final model saved: {os.path.abspath(final_path)}")

# If a best model was found by EvalCallback, it will be at:
print(f"ℹ If created, best model is: {os.path.abspath(os.path.join(OUT_DIR, 'best_model.zip'))}")



