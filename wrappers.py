# wrappers.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces

def _distance(a, b):
    return float(np.linalg.norm(a - b))

def _check_grasp(env):
    ncon = env.robot_model.data.ncon
    if ncon == 0:
        return False
    contact = env.robot_model.data.contact
    club = env.golf_club_id
    L = env.left_finger_body_id
    R = env.right_finger_body_id
    left = right = False
    for i in range(ncon):
        g1, g2 = contact[i].geom1, contact[i].geom2
        b1 = env.robot_model.model.geom_bodyid[g1]
        b2 = env.robot_model.model.geom_bodyid[g2]
        if b1 == club or b2 == club:
            other = b2 if b1 == club else b1
            if other == L: left = True
            elif other == R: right = True
    return left and right

def _check_ball_contact(env):
    ncon = env.robot_model.data.ncon
    if ncon == 0: return False
    contact = env.robot_model.data.contact
    club_head = env.club_head_id
    ball = env.golf_ball_id
    for i in range(ncon):
        g1, g2 = contact[i].geom1, contact[i].geom2
        b1 = env.robot_model.model.geom_bodyid[g1]
        b2 = env.robot_model.model.geom_bodyid[g2]
        if (b1 == club_head and b2 == ball) or (b1 == ball and b2 == club_head):
            return True
    return False

def _get_key_positions(env):
    data = env.robot_model.data
    ee_pos   = data.site(env.ee_site_id).xpos
    club_pos = data.xpos[env.golf_club_id]
    head_pos = data.xpos[env.club_head_id]
    ball_pos = data.xpos[env.golf_ball_id]
    hole_pos = data.xpos[env.golf_hole_id]
    return ee_pos.copy(), club_pos.copy(), head_pos.copy(), ball_pos.copy(), hole_pos.copy()

class RescaleAction(gym.ActionWrapper):
    """
    Map agent actions from [-1,1] to small real torques/vels.
    Using a small range prevents wild motions and helps make contact.
    """
    def __init__(self, env, min_action, max_action):
        super().__init__(env)
        self.min_action = np.array(min_action, dtype=np.float32)
        self.max_action = np.array(max_action, dtype=np.float32)
        assert self.min_action.shape == self.env.action_space.shape
        assert self.max_action.shape == self.env.action_space.shape
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=self.env.action_space.shape, dtype=np.float32)

    def action(self, act):
        act = np.clip(act, -1.0, 1.0)
        scaled = self.min_action + (act + 1.0) * 0.5 * (self.max_action - self.min_action)
        return np.clip(scaled, self.min_action, self.max_action)

class RewardShapingWrapper(gym.Wrapper):
    """
    Dense reward to speed up learning:
      - Before grasp: reduce EE→club distance, bonus on grasp
      - After grasp, before hit: reduce clubhead→ball distance, bonus on first contact
      - After first hit: reduce ball→hole distance
      - Small smoothness penalty to avoid flailing
    """
    def __init__(self, env,
                 w_ee_to_club=0.30,
                 w_grasp=1.50,
                 w_head_to_ball=0.25,
                 w_first_contact=1.00,
                 w_ball_to_hole=0.20,
                 action_smooth_penalty=0.01,
                 dist_clip=2.0):
        super().__init__(env)
        self.w_ee_to_club = w_ee_to_club
        self.w_grasp = w_grasp
        self.w_head_to_ball = w_head_to_ball
        self.w_first_contact = w_first_contact
        self.w_ball_to_hole = w_ball_to_hole
        self.action_smooth_penalty = action_smooth_penalty
        self.dist_clip = dist_clip

        self.prev = {}
        self._prev_action = None
        self._saw_grasp = False
        self._saw_ball_contact = False

    def reset(self, **kwargs):
        self.prev.clear()
        self._prev_action = None
        self._saw_grasp = False
        self._saw_ball_contact = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, env_r, terminated, truncated, info = self.env.step(action)

        ee, club, head, ball, hole = _get_key_positions(self.env.unwrapped)
        grasped   = _check_grasp(self.env.unwrapped)
        hit_ball  = _check_ball_contact(self.env.unwrapped)

        d_ee_club   = _distance(ee, club)
        d_head_ball = _distance(head, ball)
        d_ball_hole = _distance(ball, hole)

        p_ee_club   = self.prev.get("d_ee_club",   d_ee_club)
        p_head_ball = self.prev.get("d_head_ball", d_head_ball)
        p_ball_hole = self.prev.get("d_ball_hole", d_ball_hole)

        shaping = 0.0
        # Phase 1
        if not self._saw_grasp:
            shaping += self.w_ee_to_club * np.clip(p_ee_club - d_ee_club, -self.dist_clip, self.dist_clip)
            if grasped:
                self._saw_grasp = True
                shaping += self.w_grasp
        # Phase 2
        elif not self._saw_ball_contact:
            shaping += self.w_head_to_ball * np.clip(p_head_ball - d_head_ball, -self.dist_clip, self.dist_clip)
            if hit_ball:
                self._saw_ball_contact = True
                shaping += self.w_first_contact
        # Phase 3
        else:
            shaping += self.w_ball_to_hole * np.clip(p_ball_hole - d_ball_hole, -self.dist_clip, self.dist_clip)

        if self._prev_action is not None and self.action_smooth_penalty > 0:
            shaping -= self.action_smooth_penalty * float(np.linalg.norm(action - self._prev_action))
        self._prev_action = action.copy()

        info = dict(info or {})
        info["shaping"] = shaping
        info["events"] = {"grasped": self._saw_grasp, "hit_ball": self._saw_ball_contact}

        self.prev["d_ee_club"]   = d_ee_club
        self.prev["d_head_ball"] = d_head_ball
        self.prev["d_ball_hole"] = d_ball_hole

        return obs, env_r + shaping, terminated, truncated, info
