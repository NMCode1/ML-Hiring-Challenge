# evolve_sac_seeded.py
# Neuroevolution on top of your existing SAC: mutate actor weights, keep best.
import os, time, copy, argparse, math, random
import numpy as np
import torch
from stable_baselines3 import SAC
from sai_rl import SAIClient

# ---------- Config via CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--comp", default="franka-ml-hiring")
    p.add_argument("--model_path", required=True, help="Seed SAC .zip (your best_model.zip)")
    p.add_argument("--out_dir", default="./models/evo")
    p.add_argument("--pop", type=int, default=16, help="population size per generation")
    p.add_argument("--elites", type=int, default=4, help="# elites to keep")
    p.add_argument("--gens", type=int, default=10, help="# generations")
    p.add_argument("--episodes", type=int, default=5, help="episodes per candidate")
    p.add_argument("--noise", type=float, default=0.01, help="initial weight noise std")
    p.add_argument("--noise_decay", type=float, default=0.85, help="decay per gen")
    p.add_argument(
    "--risk_coeff", type=float, default=0.0,
    help="Risk penalty coefficient: rank by mean - risk_coeff * std"
)

    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()

# ---------- Deterministic action with tiny smoothing + sticky gripper ----------
def make_action_fn(model, grip_idx=6):
    last = None
    latched = 0.0
    def act(obs):
        nonlocal last, latched  # <-- add this line
        a, _ = model.predict(obs, deterministic=True)

        # Sticky gripper (hysteresis)
        g = a[grip_idx]
        if abs(g) < 0.20:
            a[grip_idx] = latched
        elif g > 0.35:
            latched = 1.0
            a[grip_idx] = latched
        elif g < -0.35:
            latched = -1.0
            a[grip_idx] = latched

        # Light low-pass filter
        if last is None:
            last = a.copy()
        a = 0.85 * last + 0.15 * a
        last = a.copy()

        return a
    return act

# ---------- Evaluate a candidate (mean score over N episodes) ----------
def eval_candidate(base_model_path, actor_state_dict, sai, episodes=5, max_steps=4000):
    # Load fresh model for clean eval
    model = SAC.load(base_model_path, device="auto")
    # Replace actor weights
    model.actor.load_state_dict(actor_state_dict, strict=True)
    act_fn = make_action_fn(model)

    env = sai.make_env()
    scores = []
    for _ in range(episodes):
        obs, info = env.reset()
        done = False; trunc = False; total = 0.0; steps = 0
        while not (done or trunc):
            a = act_fn(obs)
            obs, r, done, trunc, info = env.step(a)
            total += float(r)
            steps += 1
            if steps >= max_steps:  # safety cap
                break
        scores.append(total)
    env.close()
    return float(np.mean(scores)), float(np.std(scores))

# ---------- Mutate actor params in-place ----------
def mutate_actor(actor, noise_std):
    with torch.no_grad():
        for name, p in actor.named_parameters():
            if not p.requires_grad:
                continue
            # Smaller noise for biases; a bit larger for weights
            scale = noise_std * (0.5 if p.ndim == 1 else 1.0)
            p.add_(torch.randn_like(p) * scale)

def clone_actor_state(actor):
    return {k: v.detach().clone() for k, v in actor.state_dict().items()}

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # SAI env handle
    sai = SAIClient(comp_id=args.comp)

    # Load seed model once
    seed_model = SAC.load(args.model_path, device="auto")
    seed_actor_state = clone_actor_state(seed_model.actor)

    # Prepare initial population actor states
    def make_population(noise_std):
        pop = []
        # Elite slot 0 = exact seed (no mutation)
        pop.append(("seed", clone_actor_state(seed_model.actor)))
        # Others = mutated copies of seed
        for i in range(args.pop - 1):
            m = copy.deepcopy(seed_model.actor)
            mutate_actor(m, noise_std)
            pop.append((f"mut{i+1}", clone_actor_state(m)))
        return pop

    noise_std = args.noise
    best_overall = (-1e9, None, None)  # (mean_score, actor_state, std)

    for gen in range(1, args.gens + 1):
        print(f"\n=== Generation {gen}/{args.gens} (noise_std={noise_std:.4f}) ===")
        population = make_population(noise_std)

        # Evaluate
        results = []
        for label, actor_sd in population:
            mean, std = eval_candidate(args.model_path, actor_sd, sai, episodes=args.episodes)
            results.append((mean, std, label, actor_sd))
            print(f"  {label:>6s}  mean={mean:.3f}  std={std:.3f}")

        # Sort by mean (desc), tiebreaker lower std
        # Sort by risk-adjusted score = mean - risk_coeff * std
        results.sort(key=lambda x: (x[0] - args.risk_coeff * x[1]), reverse=True)


        # Track best overall
        if results[0][0] > best_overall[0] or (math.isclose(results[0][0], best_overall[0]) and results[0][1] < best_overall[2]):
            best_overall = (results[0][0], results[0][3], results[0][1])

        # Elites
        elites = results[:args.elites]
        elite_means = [e[0] for e in elites]
        print(f"Elites mean: {np.round(elite_means,3)}; best so far mean={best_overall[0]:.3f} std={best_overall[2]:.3f}")

        # Prepare next generation seed: average elites (optional)
        # Here we simply keep the best elite as the next seed to stay sharp.
        seed_actor_state = clone_actor_state(seed_model.actor)
        seed_model.actor.load_state_dict(elites[0][3], strict=True)

        # Save checkpoint for this generation’s best
        gen_best_path = os.path.join(args.out_dir, f"evo_gen{gen:02d}_best.zip")
        seed_model.save(gen_best_path)
        print("  ↳ saved gen-best:", gen_best_path)

        # Decay mutation noise
        noise_std *= args.noise_decay

    # Save overall best
    final_path = os.path.join(args.out_dir, f"evo_best_overall_mean{best_overall[0]:.3f}_std{best_overall[2]:.3f}.zip")
    seed_model.actor.load_state_dict(best_overall[1], strict=True)
    seed_model.save(final_path)
    print("\n✅ Evolution complete.")
    print("Best overall saved to:", final_path)

if __name__ == "__main__":
    main()
