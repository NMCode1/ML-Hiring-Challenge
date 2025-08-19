from stable_baselines3 import PPO
from sai_rl import SAIClient

## Initialize the SAI client
sai = SAIClient(comp_id="franka-golf-challenge")

## Make the environment
env = sai.make_env()

## Define the model
model = PPO("MlpPolicy", env)
model.learn(total_timesteps=100)

## Benchmark the model locally
sai.benchmark(model)

## Save and submit the model
sai.submit("My First Model", model)
