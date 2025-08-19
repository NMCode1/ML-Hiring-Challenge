# submit_deterministic.py
from sai_rl import SAIClient
from stable_baselines3 import SAC
import numpy as np

sai = SAIClient(comp_id="franka-ml-hiring")
model = SAC.load("./models/shaped_sac_safe/best_model")

def deterministic_fn(policy, obs_batch):
    outs = []
    for obs in obs_batch:
        a, _ = policy.predict(obs, deterministic=True)
        outs.append(a)
    return np.asarray(outs, dtype="float32")

result = sai.submit_model(
    name="Nick SAC (deterministic)",
    model=model,
    model_type="stable_baselines3",
    action_function=deterministic_fn,
)
print(result)
