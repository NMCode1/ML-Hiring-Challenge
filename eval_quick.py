import os, time, numpy as np
from sai_rl import SAIClient
from stable_baselines3 import SAC

# ---- scoring helpers (same as yours) ----
def distance_between_objects(pos1, pos2): return abs(np.linalg.norm(pos1 - pos2))
def ball_in_hole(ball_pos, hole_pos): return (distance_between_objects(ball_pos, hole_pos) < 0.06).astype(np.float32)

def check_grasp(env):
    ncon = env.robot_model.data.ncon
    if ncon == 0: return False
    c = env.robot_model.data.contact
    club = env.golf_club_id; L = env.left_finger_body_id; R = env.right_finger_body_id
    club_left = club_right = False
    for i in range(ncon):
        b1 = env.robot_model.model.geom_bodyid[c[i].geom1]
        b2 = env.robot_model.model.geom_bodyid[c[i].geom2]
        if b1 == club or b2 == club:
            other = b2 if b1 == club else b1
            if other == L: club_left = True
            elif other == R: club_right = True
    return club_left and club_right

def check_ball_club_contact(env):
    ncon = env.robot_model.data.ncon
    if ncon == 0: return False
    c = env.robot_model.data.contact
    club = env.club_head_id; ball = env.golf_ball_id
    for i in range(ncon):
        b1 = env.robot_model.model.geom_bodyid[c[i].geom1]
        b2 = env.robot_model.model.geom_bodyid[c[i].geom2]
        if (b1 == club and b2 == ball) or (b1 == ball and b2 == club): return True
    return False

def evaluation_fn(env, eval_state):
    if "timestep" not in eval_state: eval_state["timestep"] = 0
    if "closest_distance_to_club" not in eval_state: eval_state["closest_distance_to_club"] = 1e9

    ee_pos   = env.robot_model.data.site(env.ee_site_id).xpos
    club_pos = env.robot_model.data.xpos[env.golf_club_id]
    ball_pos = env.robot_model.data.xpos[env.golf_ball_id]
    hole_pos = env.robot_model.data.xpos[env.golf_hole_id]

    ee_club_dist = distance_between_objects(ee_pos, club_pos)
    grasp   = check_grasp(env)
    hitball = check_ball_club_contact(env)

    if hitball and "robot_hit_ball_with_club" not in eval_state:
        eval_state["robot_hit_ball_with_club"] = eval_state["timestep"]
    if grasp and "robot_grasped_club" not in eval_state:
        eval_state["robot_grasped_club"] = eval_state["timestep"]

    eval_state["closest_distance_to_club"] = min(eval_state["closest_distance_to_club"], ee_club_dist)
    eval_state["timestep"] += 1

    if eval_state.get("terminated") or eval_state.get("truncated"):
        reward = 0.0
        if ball_in_hole(ball_pos, hole_pos):
            reward += 10.0 - eval_state["timestep"] * 0.01
        else:
            reward += (1.1867 - distance_between_objects(ball_pos, hole_pos)) * 2
        if "robot_hit_ball_with_club" in eval_state:
            reward += 3.5 - eval_state["robot_hit_ball_with_club"] * 0.005
        if "robot_grasped_club" in eval_state:
            reward += 1.65 - eval_state["robot_grasped_club"] * 0.001
        else:
            reward += max(1 - eval_state["closest_distance_to_club"], 0)
        return reward, eval_state
    return 0.0, eval_state

# ---- choose which model to test ----
MODEL_PATH = "./models/evo_v3/evo_gen02_best.zip"  # change to v2 to test that one

print(">>> Loading model:", MODEL_PATH)
model = SAC.load(MODEL_PATH)  # no env needed for predict

print(">>> Creating challenge env (no extra args)...")
sai = SAIClient(comp_id="franka-ml-hiring")
env = sai.make_env()  # raw gymnasium env, closest to server config

print(">>> Resetting env...")
obs, info = env.reset(seed=0)

print(">>> Starting 1 evaluation episode (deterministic)...")
eval_state, ep_score, steps = {}, 0.0, 0

t0 = time.time()
while True:
    action, _ = model.predict(obs, deterministic=True)
    # raw Gymnasium step: obs, reward, terminated, truncated, info
    obs, reward, terminated, truncated, info = env.step(action)

    eval_state["terminated"] = bool(terminated)
    eval_state["truncated"]  = bool(truncated)

    step_score, eval_state = evaluation_fn(env.unwrapped, eval_state)
    ep_score += step_score
    steps += 1

    if terminated or truncated:
        break

dt = time.time() - t0
print(f"Done. Steps={steps}, local_score≈{ep_score:.3f}, seconds={dt:.1f}")

env.close()

