# teleop_manual.py
import os, sys, time
import numpy as np
import gymnasium as gym
from sai_rl import SAIClient

os.environ.setdefault("MUJOCO_GL", "glfw")

def clamp_action(env, a):
    return np.clip(a, env.action_space.low, env.action_space.high).astype(np.float32)

def main():
    sai = SAIClient(comp_id="franka-ml-hiring")
    # Create env with rendering; extend episode length so it doesn't end immediately
    try:
        base_env = sai.make_env(render_mode="human")
    except TypeError:
        base_env = sai.make_env(env_kwargs={"render_mode": "human"})
    env = gym.wrappers.TimeLimit(base_env, max_episode_steps=1500)

    obs, info = env.reset(seed=0)

    act_dim = int(env.action_space.shape[0])
    assert act_dim >= 7, f"Expected at least 7-dim action space, got {act_dim}"
    action = np.zeros(act_dim, dtype=np.float32)  # start stationary
    step_size = 0.02  # conservative to avoid instant terminations
    running = True
    paused = False   # when episode ends, we pause until you press 'r'

    print(f"\nAction space: {env.action_space}")
    print(f"Obs space:    {env.observation_space}")
    print("\nControls (keep THIS terminal focused while pressing keys):")
    print("  W/S : Z up/down")
    print("  A/D : X left/right")
    print("  Q/E : Y back/forward")
    print("  J/L : gripper open/close")
    print("  -/= : step size down/up")
    print("  0   : zero action vector")
    print("  r   : reset episode")
    print("  Q   : quit\n")
    print(f"step_size = {step_size:.3f}")
    print(f"Action = {np.array2string(action, precision=3, suppress_small=True)}")

    # Raw, nonblocking key input (works in macOS Terminal)
    import termios, tty, select
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    def key_available():
        return select.select([sys.stdin], [], [], 0)[0]

    step = 0
    last_reward = 0.0

    try:
        while running:
            # Handle keys
            if key_available():
                ch = sys.stdin.read(1)
                if ch:
                    c = ch  # keep case-sensitive for 'Q' quit
                    lc = c.lower()

                    # Movement (only when not paused)
                    if not paused:
                        if lc == 'w':   # Z+
                            action[2] += step_size
                            print(f"[key] Z+  action[2] += {step_size:.3f}")
                        elif lc == 's': # Z-
                            action[2] -= step_size
                            print(f"[key] Z-  action[2] -= {step_size:.3f}")
                        elif lc == 'a': # X-
                            action[0] -= step_size
                            print(f"[key] X-  action[0] -= {step_size:.3f}")
                        elif lc == 'd': # X+
                            action[0] += step_size
                            print(f"[key] X+  action[0] += {step_size:.3f}")
                        elif lc == 'q': # Y-
                            action[1] -= step_size
                            print(f"[key] Y-  action[1] -= {step_size:.3f}")
                        elif lc == 'e': # Y+
                            action[1] += step_size
                            print(f"[key] Y+  action[1] += {step_size:.3f}")
                        elif lc == 'j': # gripper open
                            action[6] += step_size
                            print(f"[key] Grip OPEN  action[6] += {step_size:.3f}")
                        elif lc == 'l': # gripper close
                            action[6] -= step_size
                            print(f"[key] Grip CLOSE action[6] -= {step_size:.3f}")

                    # Step size & utilities (work even if paused)
                    if lc == '-':
                        step_size = max(0.005, step_size * 0.5)
                        print(f"[key] step_size = {step_size:.3f}")
                    elif lc in ('=', '+'):
                        step_size = min(0.5, step_size * 2.0)
                        print(f"[key] step_size = {step_size:.3f}")
                    elif lc == '0':
                        action[:] = 0.0
                        print("[key] action zeroed")
                    elif lc == 'r':
                        obs, info = env.reset()
                        action[:] = 0.0
                        paused = False
                        step = 0
                        last_reward = 0.0
                        print("[key] reset; action=0")
                    elif c == 'Q':  # uppercase Q to quit
                        print("[key] quit")
                        break

                    # Show current action after each key
                    print(f"Action = {np.array2string(action, precision=3, suppress_small=True)}")

            # If paused, do not step until user resets
            if paused:
                time.sleep(0.02)
                continue

            # Step & render
            a = clamp_action(env, action)
            obs, reward, terminated, truncated, info = env.step(a)
            last_reward = float(reward)
            step += 1
            try:
                env.render()
            except Exception:
                pass

            if step % 10 == 0:
                print(f"[env] step={step} reward={last_reward:.3f}")

            if terminated or truncated:
                paused = True
                print("[env] episode ended ⇒ PAUSED. Press 'r' to reset (actions are preserved).")

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
