import gymnasium as gym
from sai_rl import SAIClient

# Connect to the competition
sai = SAIClient(comp_id="franka-ml-hiring")

# Create the official Franka environment
env = sai.make_env()

# Try to enable rendering
try:
    env = gym.wrappers.RecordEpisodeStatistics(env)
    obs, info = env.reset()
    print("✅ Environment created. Starting manual control test...")

    for step in range(200):
        env.render()  # Try to open a visual window
        action = env.action_space.sample()  # Random actions for now
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
except Exception as e:
    print("❌ Rendering not available:", e)

env.close()
