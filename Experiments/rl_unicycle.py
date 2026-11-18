"""
Unicycle Robot Navigation via Q-Learning (Reinforcement Learning)
==================================================================

A unicycle robot (nonholonomic kinematic model) learns to navigate from a
start position to a goal position on a 2D board while avoiding circular
obstacles. The policy is trained with tabular Q-learning on a discretized
state space, then a greedy rollout is recorded and animated. The robot is
drawn as an arrow (triangle) so its heading is always visible.

Kinematics (unicycle model):
    x'     = x + v * cos(theta) * dt
    y'     = y + v * sin(theta) * dt
    theta' = theta + omega * dt

Run this script directly with a normal Python/matplotlib install and a
window will pop up showing the training result animate live:
    python unicycle_rl.py

If you just want to (re)generate a saved GIF instead of a live window,
set SAVE_GIF = True below (or pass --save-gif on the command line).
"""
import os
import sys
import numpy as np
import matplotlib

# Check environment for displays
HAS_DISPLAY = bool(os.environ.get("DISPLAY")) or sys.platform in ("win32", "darwin")
if not HAS_DISPLAY:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches

# Configuration Imports
from config import (
    RL_BOARD_SIZE as BOARD_SIZE, RL_START as START, RL_GOAL as GOAL, RL_GOAL_RADIUS as GOAL_RADIUS,
    RL_OBSTACLES as OBSTACLES, RL_V as V, RL_DT as DT, RL_OMEGA_ACTIONS as OMEGA_ACTIONS,
    RL_N_X_BINS as N_X_BINS, RL_N_Y_BINS as N_Y_BINS, RL_N_THETA_BINS as N_THETA_BINS,
    RL_N_EPISODES as N_EPISODES, RL_MAX_STEPS_PER_EPISODE as MAX_STEPS_PER_EPISODE,
    RL_ALPHA as ALPHA, RL_GAMMA as GAMMA, RL_EPS_START as EPS_START, RL_EPS_END as EPS_END,
    RL_EPS_DECAY_EPISODES as EPS_DECAY_EPISODES, RL_SAVE_GIF as SAVE_GIF, RL_GIF_PATH as GIF_PATH,
    RL_RANDOM_SEED as RANDOM_SEED
)

rng = np.random.default_rng(RANDOM_SEED)

class UnicycleEnv:
    def __init__(self):
        self.board_size = BOARD_SIZE
        self.start = START
        self.goal = np.array(GOAL)
        self.goal_radius = GOAL_RADIUS
        self.obstacles = OBSTACLES
        self.actions = OMEGA_ACTIONS
        self.n_actions = len(self.actions)
        self.state = None
        self.prev_dist = None

    def reset(self):
        x, y, theta = self.start
        self.state = np.array([x, y, theta], dtype=float)
        self.prev_dist = np.linalg.norm(self.state[:2] - self.goal)
        return self.state.copy()

    def _collides(self, x, y):
        if x <= 0.0 or x >= self.board_size or y <= 0.0 or y >= self.board_size:
            return True
        for (ox, oy, r) in self.obstacles:
            if (x - ox) ** 2 + (y - oy) ** 2 <= r ** 2:
                return True
        return False

    def step(self, action_idx):
        x, y, theta = self.state
        omega = self.actions[action_idx]
        new_theta = theta + omega * DT
        new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi
        new_x = x + V * np.cos(theta) * DT
        new_y = y + V * np.sin(theta) * DT

        if self._collides(new_x, new_y):
            reward = -50.0
            done = True
            self.state = np.array([x, y, new_theta])
            return self.state.copy(), reward, done

        self.state = np.array([new_x, new_y, new_theta])
        dist = np.linalg.norm(self.state[:2] - self.goal)
        reached_goal = dist <= self.goal_radius
        progress = self.prev_dist - dist
        reward = -0.1 + 2.0 * progress
        self.prev_dist = dist
        done = False
        if reached_goal:
            reward += 100.0
            done = True
        return self.state.copy(), reward, done

def discretize(state):
    x, y, theta = state
    xi = int(np.clip(x / BOARD_SIZE * N_X_BINS, 0, N_X_BINS - 1))
    yi = int(np.clip(y / BOARD_SIZE * N_Y_BINS, 0, N_Y_BINS - 1))
    theta_norm = (theta + np.pi) / (2 * np.pi)
    ti = int(np.clip(theta_norm * N_THETA_BINS, 0, N_THETA_BINS - 1))
    return xi, yi, ti

class QLearningAgent:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.Q = np.zeros((N_X_BINS, N_Y_BINS, N_THETA_BINS, n_actions))

    def choose_action(self, s_idx, epsilon):
        if rng.random() < epsilon:
            return rng.integers(0, self.n_actions)
        return int(np.argmax(self.Q[s_idx]))

    def update(self, s_idx, a, reward, s_next_idx, done):
        best_next = 0.0 if done else np.max(self.Q[s_next_idx])
        td_target = reward + GAMMA * best_next
        td_error = td_target - self.Q[s_idx][a]
        self.Q[s_idx][a] += ALPHA * td_error

def epsilon_by_episode(ep):
    frac = min(ep / EPS_DECAY_EPISODES, 1.0)
    return EPS_START + frac * (EPS_END - EPS_START)

def train():
    env = UnicycleEnv()
    agent = QLearningAgent(env.n_actions)
    success_window = []

    for ep in range(N_EPISODES):
        state = env.reset()
        s_idx = discretize(state)
        epsilon = epsilon_by_episode(ep)
        total_reward = 0.0

        for _ in range(MAX_STEPS_PER_EPISODE):
            a = agent.choose_action(s_idx, epsilon)
            next_state, reward, done = env.step(a)
            s_next_idx = discretize(next_state)
            agent.update(s_idx, a, reward, s_next_idx, done)
            state, s_idx = next_state, s_next_idx
            total_reward += reward
            if done:
                break

        reached = np.linalg.norm(state[:2] - env.goal) <= env.goal_radius
        success_window.append(1 if reached else 0)
        if len(success_window) > 200:
            success_window.pop(0)

        if (ep + 1) % 500 == 0:
            rate = 100.0 * sum(success_window) / len(success_window)
            print(f"Episode {ep + 1:5d}/{N_EPISODES} | eps={epsilon:.3f} | reward={total_reward:7.2f} | recent success rate={rate:5.1f}%")
    return env, agent

def rollout(env, agent, max_steps=300):
    state = env.reset()
    trajectory = [state.copy()]
    for _ in range(max_steps):
        s_idx = discretize(state)
        a = int(np.argmax(agent.Q[s_idx]))
        state, _, done = env.step(a)
        trajectory.append(state.copy())
        if done:
            break
    reached = np.linalg.norm(state[:2] - env.goal) <= env.goal_radius
    return np.array(trajectory), reached

def make_arrow_vertices(length=0.55, width=0.30):
    return np.array([[length * 0.65, 0.0], [-length * 0.35, width * 0.5], [-length * 0.35, -width * 0.5]])

def transform(local_pts, x, y, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return (R @ local_pts.T).T + np.array([x, y])

def animate_trajectory(env, trajectory, reached, save_gif=False, gif_path=GIF_PATH):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, env.board_size)
    ax.set_ylim(0, env.board_size)
    ax.set_aspect("equal")
    ax.set_title("Unicycle Robot Navigation (Q-Learning)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    for (ox, oy, r) in env.obstacles:
        ax.add_patch(mpatches.Circle((ox, oy), r, color="dimgray", alpha=0.7, zorder=2))

    ax.plot(env.start[0], env.start[1], "go", markersize=10, label="Start", zorder=3)
    ax.add_patch(mpatches.Circle(env.goal, env.goal_radius, color="gold", alpha=0.4, zorder=2))
    ax.plot(env.goal[0], env.goal[1], "y*", markersize=18, label="Goal", zorder=3)

    ax.plot(trajectory[:, 0], trajectory[:, 1], "--", color="royalblue", alpha=0.25, linewidth=1.5, zorder=1)
    trail_line, = ax.plot([], [], "-", color="royalblue", linewidth=2, zorder=3)

    local_pts = make_arrow_vertices()
    x0, y0, th0 = trajectory[0]
    robot_patch = mpatches.Polygon(transform(local_pts, x0, y0, th0), closed=True, color="crimson", zorder=4)
    ax.add_patch(robot_patch)

    status_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize=10, bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.legend(loc="lower right")

    def init():
        trail_line.set_data([], [])
        robot_patch.set_xy(transform(local_pts, x0, y0, th0))
        status_text.set_text("")
        return trail_line, robot_patch, status_text

    def update(frame):
        x_v, y_v, theta_v = trajectory[frame]
        trail_line.set_data(trajectory[:frame + 1, 0], trajectory[:frame + 1, 1])
        robot_patch.set_xy(transform(local_pts, x_v, y_v, theta_v))
        status_text.set_text(f"step {frame}/{len(trajectory) - 1}\npos=({x_v:.2f}, {y_v:.2f})  heading={np.degrees(theta_v):.0f} deg")
        return trail_line, robot_patch, status_text

    anim = animation.FuncAnimation(fig, update, frames=len(trajectory), init_func=init, interval=80, blit=False, repeat=True)
    outcome = "reached the goal! \N{PARTY POPPER}" if reached else "did not reach the goal in time."
    print(f"Rollout finished: the robot {outcome}")

    if save_gif:
        out_dir = os.path.dirname(os.path.abspath(gif_path))
        os.makedirs(out_dir, exist_ok=True)
        anim.save(gif_path, writer="pillow", fps=12)
        print(f"Saved animation to {os.path.abspath(gif_path)}")

    if HAS_DISPLAY:
        plt.show()
    else:
        plt.close(fig)
    return anim

if __name__ == "__main__":
    save_gif_flag = SAVE_GIF or ("--save-gif" in sys.argv)
    print("Training the unicycle robot with Q-learning...")
    env, agent = train()
    print("\nGenerating greedy rollout trajectory...")
    trajectory, reached = rollout(env, agent, max_steps=MAX_STEPS_PER_EPISODE)
    print(f"Trajectory length: {len(trajectory)} steps")
    print("\nAnimating...")
    animate_trajectory(env, trajectory, reached, save_gif=save_gif_flag)