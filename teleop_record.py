# teleop_record.py
import os, sys, time, datetime
import numpy as np
import gymnasium as gym
from sai_rl import SAIClient

os.environ.setdefault("MUJOCO_GL", "glfw")

def clamp_action(env, a):
    return np.clip(a, env.action_space.low, env.action_space.high).astype(np.float32)

def now_stamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def main():
    sai = SAIClient(comp_id="franka-ml-hiring")
    # Viewer + longer episode
    try:
        base_env = sai.make_env(render_mode="human")
    except TypeError:
        base_env = sai.make_env(env_kwargs={"render_mode": "human"})
    env = gym.wrappers.TimeLimit(base_env, max_episode_steps=1500)

    os.makedirs("demos", exist_ok=True)

    obs, info = env.reset(seed=0)
    act_dim = int(env.action_space.shape[0])
    assert act_dim >= 7, f"Expected at least 7-dim action space, got {act_dim}"
    action = np.zeros(act_dim, dtype=np.float32)
    step_size = 0.02
    paused = False
    running = True

    # Recording buffers (session-wide)
    obs_buf, act_buf, rew_buf, done_buf = [], [], [], []
    recording = True  # toggle with 't'

    print(f"\nAction space: {env.action_space}")
    print(f"Obs space:    {env.observation_space}")
    print("\nControls (Terminal must stay focused):")
    print("  W/S : Z up/down       |  A/D : X left/right     |  Q/E : Y back/forward")
    print("  J/L : gripper open/close")
    print("  -/= : step size down/up |  0 : zero action")
    print("  r   : reset episode     |  t : toggle recording (on by default)")
    print("  p   : save buffer to demos/*.npz (you can save multiple times)")
    print("  Q   : quit (auto-save)")
    print(f"\nRecording = {recording} | step_size = {step_size:.3f}")
    print(f"Action = {np.array2string(action, precision=3, suppress_small=True)}")

    # Nonblocking key input (macOS Terminal)
    import termios, tty, select
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    def key_available():
        return select.select([sys.stdin], [], [], 0)[0]

    def save_now():
        if not obs_buf:
            print("[save] buffer empty; nothing to save.")
            return
        fname = f"demos/teleop_{now_stamp()}.npz"
        np.savez_compressed(
            fname,
            obs=np.array(obs_buf, dtype=np.float32),
            act=np.array(act_buf, dtype=np.float32),
            rew=np.array(rew_buf, dtype=np.float32),
            done=np.array(done_buf, dtype=np.bool_),
        )
        print(f"[save] wrote {fname}  steps={len(obs_buf)}  episodes≈{int(np.sum(done_buf))}")

    step = 0
    ep_steps = 0
    last_reward = 0.0

    try:
        while running:
            # --- Key handling ---
            if key_available():
                ch = sys.stdin.read(1)
                if ch:
                    c = ch  # keep case-sensitive for 'Q' quit
                    lc = c.lower()

                    # Movement (disabled when paused)
                    if not paused:
                        if lc == 'w':   action[2] += step_size; print(f"[key] Z+  a[2]+={step_size:.3f}")
                        elif lc == 's': action[2] -= step_size; print(f"[key] Z-  a[2]-={step_size:.3f}")
                        elif lc == 'a': action[0] -= step_size; print(f"[key] X-  a[0]-={step_size:.3f}")
                        elif lc == 'd': action[0] += step_size; print(f"[key] X+  a[0]+={step_size:.3f}")
                        elif lc == 'q': action[1] -= step_size; print(f"[key] Y-  a[1]-={step_size:.3f}")
                        elif lc == 'e': action[1] += step_size; print(f"[key] Y+  a[1]+={step_size:.3f}")
                        elif lc == 'j': action[6] += step_size; print(f"[key] Grip OPEN  a[6]+={step_size:.3f}")
                        elif lc == 'l': action[6] -= step_size; print(f"[key] Grip CLOSE a[6]-={step_size:.3f}")

                    # Utilities
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
                        ep_steps = 0
                        last_reward = 0.0
                        print("[key] reset; action=0")
                    elif lc == 't':
                        recording = not recording
                        print(f"[key] recording = {recording}")
                    elif lc == 'p':
                        save_now()
                    elif c == 'Q':  # uppercase Q quits
                        print("[key] quit (auto-save)")
                        save_now()
                        break

                    # Show current action after each key
                    print(f"Action = {np.array2string(action, precision=3, suppress_small=True)}")

            # If paused, don't step until reset
            if paused:
                time.sleep(0.02)
                continue

            # --- Env step ---
            a = clamp_action(env, action)
            next_obs, reward, terminated, truncated, info = env.step(a)
            try:
                env.render()
            except Exception:
                pass_

            # Record this step
            if recording:
                obs_buf.append(obs.copy())
                act_buf.append(a.copy())
                rew_buf.append(float(reward))
                done_buf.append(bool(terminated or truncated))

            obs = next_obs
            step += 1
            ep_steps += 1
            last_reward = float(reward)

            if step % 20 == 0:
                print(f"[env] step={step} (session), ep_steps={ep_steps}, reward={last_reward:.3f}, rec_steps={len(obs_buf)}")

            if terminated or truncated:
                paused = True
                print(f"[env] episode ended after {ep_steps} steps ⇒ PAUSED. Press 'r' to reset. (rec={len(obs_buf)})")

            time.sleep(0.02)  # ~50 FPS

    finally:
        # Restore terminal & close env
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            pass
        env.close()
        print("\nClosed.")

if __name__ == "__main__":
    main()
