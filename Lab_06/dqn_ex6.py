import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# Set up environment
env = gym.make("MountainCar-v0")
n_actions = env.action_space.n
state_dim = env.observation_space.shape[0]

# Hyperparameters
gamma = 0.99
alpha = 0.001
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
num_episodes = 500
batch_size = 64
replay_buffer_size = 100_000
target_update_freq = 10

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")

# Define Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize Q-network and optimizer
q_net = QNetwork(state_dim, n_actions).to(device)
target_net = QNetwork(state_dim, n_actions).to(device)
target_net.load_state_dict(q_net.state_dict())
target_net.eval()

optimizer = optim.Adam(q_net.parameters(), lr=alpha)
loss_fn = nn.MSELoss()
replay_buffer = deque(maxlen=replay_buffer_size)

# Epsilon-greedy
def epsilon_greedy(state, eps):
    if random.random() < eps:
        return env.action_space.sample()
    with torch.no_grad():
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = q_net(state_t)
        return torch.argmax(q_values, dim=1).item()

# Train step (experience replay)
def train_dqn():
    if len(replay_buffer) < batch_size:
        return
    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    # stack cleanly; all entries are plain obs arrays (not (obs, info))
    states      = torch.as_tensor(np.stack(states), dtype=torch.float32, device=device)
    actions     = torch.as_tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
    rewards     = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    next_states = torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=device)
    dones       = torch.as_tensor(dones, dtype=torch.float32, device=device)

    # Q(s,a)
    q_vals = q_net(states).gather(1, actions).squeeze(1)  # (B,)

    # Target: r + gamma * max_a' Q_target(s', a') * (1 - done)
    with torch.no_grad():
        next_q = target_net(next_states).max(1)[0]
        targets = rewards + gamma * next_q * (1.0 - dones)

    loss = loss_fn(q_vals, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# MAIN loop
rewards_dqn = []
steps_done = 0

for episode in range(num_episodes):
    state, _ = env.reset()  # <-- FIX: unpack (obs, info), store ONLY obs
    total_reward = 0.0
    done = False

    while not done:
        action = epsilon_greedy(state, epsilon)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # store ONLY obs arrays
        replay_buffer.append((state, action, reward, next_state, float(done)))

        state = next_state
        total_reward += reward

        train_dqn()

        if steps_done > 0 and steps_done % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())
        steps_done += 1

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards_dqn.append(total_reward)

    print("Reward: ", total_reward)

# Plot
plt.plot(rewards_dqn)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN on MountainCar-v0")
plt.grid()
plt.show()
