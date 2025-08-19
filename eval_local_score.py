# eval_local_score.py
# Local leaderboard estimator for SAI ML Hiring (Franka Golf), no extra env args.
# Seeding is applied via VecEnv.seed(...) after creation to avoid config override warnings.

import os
import numpy as np

from sai_rl import SAIClient
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    import sai_mujoco  # ensure envs register
except Exception:
    pass

# ========== Your evaluation function ==========
_FLOAT_EPS = np.finfo(np.float64).eps

def distance_between_objects(pos1, pos2):
    return abs(np.linalg.norm(pos1 - pos2))

def ball_in_hole(ball_pos, hole_pos):
    return (distance_between_objects(ball_pos, hole_pos) < 0.06).astype(np.float32)

def check_grasp(env):
    club_body_id = env.golf_club_id
    left_finger_body_id = env.left_finger_body_id
    right_finger_body_id = env.right_finger_body_id
    ncon = env.robot_model.data.ncon
    if ncon == 0:
        return False
    contact = env.robot_model.data.contact
    club_left_contact = False
    club_right_contact = False
    for i in range(ncon):
        geom1 = contact[i].geom1
        geom2 = contact[i].geom2
        body1 = env.robot_model.model.geom_bodyid[geom1]
        body2 = env.robot_model.model.geom_bodyid[geom2]
        if body1 == club_body_id or body2 == club_body_id:
            other_body = body2 if body1 == club_body_id else body1
            if other_body == left_finger_body_id:
                club_left_contact = True
            elif other_body == right_finger_body_id:
                club_right_contact = True
    return club_left_contact and club_right_contact

def check_ball_club_contact(env):
    club_body_id = env.club_head_id
    ball_body_id = env.golf_ball_id
    ncon = env.robot_model.data.ncon
    if ncon == 0:
        return False
    contact = env.robot_model.data.contact
    for i in range(ncon):
        geom1 = contact[i].geom1
        geom2 = contact[i].geom2
        body1 = env.robot_model.model.geom_bodyid[geom1]
        body2 = env.robot_model.model.geom_bodyid[geom2]
        if (body1 == club_body_id and body2 == ball_body_id) or (
            body1 == ball_body_id and body2 == club_body_id
        ):
            return True
    return False

def evaluation_fn(env, eval_state):
    if not eval_state.get("timestep", False):
        eval_state["timestep"] = 0
    if not eval_state.get("closest_distance_to_club", False):
        eval_state["closest_distance_to_club"] = 10000

    ee_pos = env.robot_model.data.site(env.ee_site_id).xpos
    club_grip_pos = env.robot_model.data.xpos[env.golf_club_id]
    ball_pos = env.robot_model.data.xpos[env.golf_ball_id]
    hole_pos = env.robot_model.data.xpos[env.golf_hole_id]

    ee_club_dist = distance_between_objects(ee_pos, club_grip_pos)
    robot_grasped_club = check_grasp(env)
    robot_hit_ball_with_club = check_ball_club_contact(env)

    if robot_hit_ball_with_club and not eval_state.get("robot_hit_ball_with_club", False):
        eval_state["robot_hit_ball_with_club"] = eval_state["timestep"]
    if robot_grasped_club and not eval_state.get("robot_grasped_club", False):
        eval_state["robot_grasped_club"] = eval_state["timestep"]

    eval_state["closest_distance_to_club"] = min(
        eval_state["closest_distance_to_club"], ee_club_dist
    )
    eval_state["timestep"] += 1

    if eval_state.get("terminated", False) or eval_state.get("truncated", False):
        reward = 0.0
        if ball_in_hole(ball_pos, hole_pos):
            reward += (10.0 - eval_state["timestep"] * 0.01)
        else:
            reward += (1.1867 - distance_between_objects(ball_pos, hole_pos)) * 2
        if eval_state.get("robot_hit_ball_with_club", False):
            reward += (3.5 - eval_state["robot_hit_ball_with_club"] * 0.005)
        if eval_state.get("robot_grasped_club", False):
            reward += (1.65 - eval_state["robot_grasped_club"] * 0.001)
        else:
            reward += max(1 - eval_state["closest_distance_to_club"], 0)
        return (reward, eval_state)
    return (0.0, eval_state)

# ========== Eval helper ==========
def eval_model(model_path, episodes=50, seed=0):
    if not os.path.exists(model_path):
        return None, f"Missing file: {model_path}"

    sai = SAIClient(comp_id="franka-ml-hiring")

    def make_env():
        # IMPORTANT: no kwargs here → matches competition config
        return Monitor(sai.make_env())

    env = DummyVecEnv([make_env])

    # Seed AFTER creation to avoid override warning
    try:
        env.seed(seed)
    except Exception:
        pass  # older/newer SB3 may not expose seed(); it's fine to proceed unseeded

    try:
        model = SAC.load(model_path, env=env)
    except Exception as e:
        env.close()
        return None, f"Load error for {model_path}: {e!r}"

    scores = []
    for ep in range(episodes):
        obs = env.reset()                  # VecEnv: reset() -> obs only
        # Also seed the underlying env on first step if method exists
        try:
            env.env_method("reset", seed=seed)
        except Exception:
            pass

        eval_state = {}
        done = [False]
        ep_score = 0.0

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)  # VecEnv: 4 returns

            info0 = infos[0]
            time_limit_trunc = bool(info0.get("TimeLimit.truncated", False))
            # Heuristic terminal flags compatible with SB3 VecEnv
            eval_state["terminated"] = bool(dones[0] and not time_limit_trunc) or bool(info0.get("terminal_observation") is not None)
            eval_state["truncated"]  = bool(time_limit_trunc)

            raw_env = env.envs[0].env.unwrapped
            step_score, eval_state = evaluation_fn(raw_env, eval_state)
            ep_score += step_score

            done = [bool(dones[0])]
        scores.append(ep_score)

    env.close()
    scores = np.array(scores, dtype=np.float32)
    return (float(scores.mean()), float(scores.std())), None

# ========== Run both v2 and v3 ==========
if __name__ == "__main__":
    MODELS = [
        ("v2", "./models/ml_hiring_sac_v2/best_model.zip"),
        ("v3", "./models/ml_hiring_sac_v3/best_model.zip"),
    ]
    EPISODES = int(os.environ.get("EVAL_EPISODES", "50"))
    SEED = int(os.environ.get("EVAL_SEED", "0"))

    print(f"Evaluating {len(MODELS)} model(s) over {EPISODES} episodes (seed={SEED})...\n")

    best_name, best_mean = None, -1e9
    for name, path in MODELS:
        result, err = eval_model(path, episodes=EPISODES, seed=SEED)
        if err:
            print(f"[{name}] ❌ {err}")
            continue
        mean, std = result
        print(f"[{name}] Local score estimate: {mean:.3f} ± {std:.3f}  (N={EPISODES})")
        if mean > best_mean:
            best_mean, best_name = mean, name

    if best_name is not None:
        print(f"\n✅ Highest local estimate: {best_name} at {best_mean:.3f}")
    else:
        print("\n⚠️ No models evaluated (check paths).")
