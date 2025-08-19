# eval_multi.py
# Multi-episode local scorer for SAI ML Hiring (Franka Golf).
# Uses raw Gymnasium env loop (closest to server behavior).

import os, time, argparse, numpy as np
from sai_rl import SAIClient
from stable_baselines3 import SAC

# ---------- scoring helpers ----------
def distance_between_objects(p1, p2): return abs(np.linalg.norm(p1 - p2))
def ball_in_hole(ball_pos, hole_pos): return (distance_between_objects(ball_pos, hole_pos) < 0.06).astype(np.float32)

def check_grasp(env):
    ncon = env.robot_model.data.ncon
    if ncon == 0: return False
    c = env.robot_model.data.contact
    club = env.golf_club_id; L = env.left_finger_body_id; R = env.right_finger_body_id
    left = right = False
    for i in range(ncon):
        b1 = env.robot_model.model.geom_bodyid[c[i].geom1]
        b2 = env.robot_model.model.geom_bodyid[c[i].geom2]
        if b1 == club or b2 == club:
            other = b2 if b1 == club else b1
            if other == L: left = True
            elif other == R: right = True
    return left and right

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

def evaluation_fn(env, st):
    if "timestep" not in st: st["timestep"] = 0
    if "closest_distance_to_club" not in st: st["closest_distance_to_club"] = 1e9

    ee_pos   = env.robot_model.data.site(env.ee_site_id).xpos
    club_pos = env.robot_model.data.xpos[env.golf_club_id]
    ball_pos = env.robot_model.data.xpos[env.golf_ball_id]
    hole_pos = env.robot_model.data.xpos[env.golf_hole_id]

    ee_club_dist = distance_between_objects(ee_pos, club_pos)
    grasp   = check_grasp(env)
    hitball = check_ball_club_contact(env)

    if hitball and "robot_hit_ball_with_club" not in st: st["robot_hit_ball_with_club"] = st["timestep"]
    if grasp and "robot_grasped_club"      not in st: st["robot_grasped_club"]      = st["timestep"]

    st["closest_distance_to_club"] = min(st["closest_distance_to_club"], ee_club_dist)
    st["timestep"] += 1

    if st.get("terminated") or st.get("truncated"):
        reward = 0.0
        if ball_in_hole(ball_pos, hole_pos):
            reward += 10.0 - st["timestep"] * 0.01
            st["ball_in_hole"] = st.get("ball_in_hole", st["timestep"])
        else:
            reward += (1.1867 - distance_between_objects(ball_pos, hole_pos)) * 2
        if "robot_hit_ball_with_club" in st:
            reward += 3.5 - st["robot_hit_ball_with_club"] * 0.005
        if "robot_grasped_club" in st:
            reward += 1.65 - st["robot_grasped_club"] * 0.001
        else:
            reward += max(1 - st["closest_distance_to_club"], 0)
        return reward, st
    return 0.0, st

# ---------- eval loop ----------
def run_eval(model_path, episodes=50, seed=0):
    sai = SAIClient(comp_id="franka-ml-hiring")
    model = SAC.load(model_path)

    scores = []
    times_grasp, times_hit, times_hole = [], [], []

    for ep in range(episodes):
        env = sai.make_env()       # no extra kwargs → matches comp config
        obs, info = env.reset(seed=seed + ep)  # seed via reset (no override warning)
        st, ep_score = {}, 0.0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            st["terminated"] = bool(terminated); st["truncated"] = bool(truncated)
            step_score, st = evaluation_fn(env.unwrapped, st)
            ep_score += step_score
            if terminated or truncated:
                scores.append(ep_score)
                if "robot_grasped_club" in st:      times_grasp.append(st["robot_grasped_club"])
                if "robot_hit_ball_with_club" in st: times_hit.append(st["robot_hit_ball_with_club"])
                if "ball_in_hole" in st:            times_hole.append(st["ball_in_hole"])
                break
        env.close()

    scores = np.array(scores, dtype=np.float32)
    def summarize(times):
        if len(times)==0: return (0.0, None)
        arr = np.array(times, dtype=np.float32)
        return (len(arr)/episodes, float(arr.mean()))
    grasp_rate, grasp_time = summarize(times_grasp)
    hit_rate,   hit_time   = summarize(times_hit)
    hole_rate,  hole_time  = summarize(times_hole)

    return {
        "mean": float(scores.mean()),
        "std":  float(scores.std()),
        "grasp_rate": grasp_rate, "grasp_time_avg": grasp_time,
        "hit_rate":   hit_rate,   "hit_time_avg":   hit_time,
        "hole_rate":  hole_rate,  "hole_time_avg":  hole_time,
        "episodes": episodes
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./models/ml_hiring_sac_v3/best_model.zip")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    stats = run_eval(args.model, episodes=args.episodes, seed=args.seed)
    dt = time.time() - t0

    print(f"\nModel: {args.model}")
    print(f"Local score: {stats['mean']:.3f} ± {stats['std']:.3f} (N={stats['episodes']})")
    print(f"Grasp: {stats['grasp_rate']*100:.1f}%   avg_t={stats['grasp_time_avg']}")
    print(f"Hit:   {stats['hit_rate']*100:.1f}%   avg_t={stats['hit_time_avg']}")
    print(f"Hole:  {stats['hole_rate']*100:.1f}%   avg_t={stats['hole_time_avg']}")
    print(f"Took {dt:.1f}s")



