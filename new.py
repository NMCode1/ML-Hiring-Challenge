from stable_baselines3 import SAC
from sai_rl import SAIClient

MODEL = "./models/evo_v3/evo_gen01_best.zip"

def make_action_fn(model, grip_idx=6):
    last = None
    latched = 0.0
    def act(obs):
        nonlocal last, latched
        a, _ = model.predict(obs, deterministic=True)
        g = a[grip_idx]
        if abs(g) < 0.20:
            a[grip_idx] = latched
        elif g > 0.35:
            latched = 1.0;  a[grip_idx] = latched
        elif g < -0.35:
            latched = -1.0; a[grip_idx] = latched
        if last is None:
            last = a.copy()
        a = 0.85 * last + 0.15 * a
        last = a.copy()
        return a
    return act

# Create environment
sai = SAIClient(comp_id="franka-ml-hiring")
env = sai.make_env(render_mode="human")  # "human" should pop up a viewer window if supported

# Load model
model = SAC.load(MODEL, device="auto")
act_fn = make_action_fn(model)

# Run ONE episode visually
obs, info = env.reset()
done = False
trunc = False
total = 0.0

while not (done or trunc):
    env.render()  # show the frame
    a = act_fn(obs)
    obs, r, done, trunc, info = env.step(a)
    total += float(r)

env.close()
print(f"Episode total score = {total:.3f}")
