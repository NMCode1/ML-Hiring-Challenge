import gym
from stable_baselines3 import SAC
from wrappers import RescaleAction, ObsNormWrapper, RewardShapingWrapper

env = gym.make(sai.make_env())
env = RescaleAction(env, [-0.25]*env.action_space.shape[0], [0.25]*env.action_space.shape[0])
obs_norm = ObsNormWrapper(env)
env = RewardShapingWrapper(obs_norm)

model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    batch_size=1024,
    learning_rate=3e-4,
    buffer_size=1_000_000,
    gamma=0.99,
    tau=0.005,
    train_freq=1,
    gradient_steps=1,
    target_update_interval=1,
    ent_coef="auto",
)

model.learn(total_timesteps=1_000_000)
obs_norm.save_stats("grasp_obs_stats.npz")
model.save("grasp_policy")
