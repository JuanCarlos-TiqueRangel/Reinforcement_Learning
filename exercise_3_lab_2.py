import numpy as np
import gymnasium as gym
import gym_simplegrid

# ----- Environment -----
env = gym.make("SimpleGrid-8x8-v0", render_mode="human")
START_LOC, GOAL_LOC = 15, 3
options = {"start_loc": START_LOC, "goal_loc": GOAL_LOC}

grid_size = 8
nS = grid_size * grid_size
gamma = 1.0

# ----- MC value estimates -----
V = np.zeros(nS)
returns_sum = np.zeros(nS)
returns_count = np.zeros(nS)
episodes = 2000      # number of training episodes

for ep in range(episodes):
    obs, info = env.reset(seed=ep, options=options)
    episode = []              # [(state, reward), ...]
    done = False

    while not done:
        # ---- choose random action and step ----
        action = env.action_space.sample()
        next_obs, reward, done, _, info = env.step(action)

        # record step and force the window to update
        episode.append((obs, reward))
        obs = next_obs
        env.render()          # keep the animation live

    # ---- Monte Carlo return computation ----
    G = 0
    visited = set()
    for t in reversed(range(len(episode))):
        s, r = episode[t]
        G = gamma * G + r
        if s not in visited:
            visited.add(s)
            returns_sum[s] += G
            returns_count[s] += 1
            V[s] = returns_sum[s] / returns_count[s]

env.close()

# reshape and print final V
print("Estimated V(s) after Monte Carlo:")
print(np.round(V.reshape(grid_size, grid_size), 1))
