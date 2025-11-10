import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# Create the CartPole environment
env = gym.make("CartPole-v1")

# Neural network model for approximating Q-values
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Hyperparameters
learning_rate = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
target_update_freq = 1000     # by steps
memory_size = 10000
episodes = 100

# Initialize Q-networks
# (Gymnasium spaces are the same shapes you expect)
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
policy_net = DQN(input_dim, output_dim)
target_net = DQN(input_dim, output_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
memory = deque(maxlen=memory_size)

# Function to choose action using epsilon-greedy policy
def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()  # Explore
    else:
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = policy_net(state_t)
            return torch.argmax(q_values, dim=1).item()  # Exploit

# Function to optimize the model using experience replay
def optimize_model():
    if len(memory) < batch_size:
        return
    
    batch = random.sample(memory, batch_size)
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

    state_batch      = torch.as_tensor(np.array(state_batch), dtype=torch.float32)
    action_batch     = torch.as_tensor(action_batch, dtype=torch.long).unsqueeze(1)
    reward_batch     = torch.as_tensor(reward_batch, dtype=torch.float32)
    next_state_batch = torch.as_tensor(np.array(next_state_batch), dtype=torch.float32)
    done_batch       = torch.as_tensor(done_batch, dtype=torch.float32)

    # Compute Q-values for current states
    q_values = policy_net(state_batch).gather(1, action_batch).squeeze(1)

    # Compute target Q-values using the target network
    with torch.no_grad():
        max_next_q_values = target_net(next_state_batch).max(1)[0]
        target_q_values = reward_batch + gamma * max_next_q_values * (1.0 - done_batch)

    loss = nn.MSELoss()(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Main training loop
rewards_per_episode = []
steps_done = 0

for episode in range(episodes):
    state, _ = env.reset()   # Gymnasium: reset() -> (obs, info)
    episode_reward = 0.0
    done = False
    
    while not done:
        # Select action
        action = select_action(state, epsilon)

        # Gymnasium: step() -> (obs, reward, terminated, truncated, info)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store transition in memory
        memory.append((state, action, reward, next_state, float(done)))
        
        # Update state
        state = next_state
        episode_reward += reward
        
        # Optimize model
        optimize_model()

        # Update target network periodically
        if steps_done > 0 and steps_done % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        steps_done += 1

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon_decay * epsilon)
    rewards_per_episode.append(episode_reward)

# Plotting the rewards per episode
plt.plot(rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('DQN on CartPole')
plt.show()
