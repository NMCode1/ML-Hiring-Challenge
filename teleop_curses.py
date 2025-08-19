# teleop_simple.py
import os, sys, time
import numpy as np
from sai_rl import SAIClient

os.environ.setdefault("MUJOCO_GL", "glfw")

def clamp_action(env, a):
    return np.clip(a, env.action_space.low, env.action_space.high).astype(np.float32)

def main():
    sai = SAIClient(comp_id="franka-ml-hiring")
    try:
        env = sai.make_env(render_mode="human")
    except TypeError:
        env = sai.make_env(env_kwargs={"render_mode": "human"})

    obs, info = env.reset(seed=0)

    act_dim = int(env.action_space.shape[0])
    action = np.zeros(act_dim, dtype=np.float32)

    # start with visible motion so you KNOW stepping works
    action[:] = np.minimum(env.action_space.high, 0.3).astype(np.float32)
    idx = 0
    step_size = 0.05

    print(f"\nAction space: {env.action_space}")
    print(f"Obs space:    {env.observation_space}")
    print("\nControls (FOCUS THIS TERMINAL while pressing keys):")
    print("  A/D : select action index (0..6)")
    print("  J/L : nudge current index [-/+] by step_size")
    print("  -/= : decrease/increase step_size")
    print("  0   : zero the whole action vector")
    print("  R   : reset episode")
    print("  Q   : quit\n")

    print(f"Editing index = {idx}, step_size = {step_size:.3f}")
    print(f"Action = {np.array2string(action, precision=3, suppress_small=True)}")

    try:
        import termios, tty, select
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)  # nonblocking raw-ish mode

        def key_available():
            return select.select([sys.stdin], [], [], 0)[0]

        step = 0
        while True:
            # read key if available
            if key_available():
                ch = sys.stdin.read(1)
                if ch:
                    c = ch.lower()
                    if c == 'a':
                        idx = (idx - 1) % act_dim
                        print(f"[key] idx <- {idx}")
                    elif c == 'd':
                        idx = (idx + 1) % act_dim
                        print(f"[key] idx -> {idx}")
                    elif c == 'j':
                        action[idx] -= step_size
                        print(f"[key] action[{idx}] -= {step_size:.3f}")
                    elif c == 'l':
                        action[idx] += step_size
                        print(f"[key] action[{idx}] += {step_size:.3f}")
                    elif c == '-':
                        step_size = max(0.005, step_size * 0.5)
                        print(f"[key] step_size = {step_size:.3f}")
                    elif c in ('=', '+'):
                        step_size = min(0.5, step_size * 2.0)
                        print(f"[key] step_size = {step_size:.3f}")
                    elif c == '0':
                        action[:] = 0
                        print("[key] action zeroed")
                    elif c == 'r':
                        obs, info = env.reset()
                        action[:] = 0.2  # small non-zero so you see motion after reset
                        print("[key] reset; action set to small non-zero")
                    elif c == 'q':
                        print("[key] quit")
                        break

                    print(f"Action = {np.array2string(action, precision=3, suppress_small=True)}")

            # step & render
            a = clamp_action(env, action)
            obs, reward, terminated, truncated, info = env.step(a)
            try:
                env.render()
            except Exception:
                pass

            if terminated or truncated:
                obs, info = env.reset()
                # keep small non-zero so you still see motion after resets
                action[:] = 0.2
                print("[env] episode ended; auto-reset; action set to 0.2")

            step += 1
            time.sleep(0.02)  # ~50 FPS pacing

    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            pass
        env.close()
        print("\nClosed.")

if __name__ == "__main__":
    main()
