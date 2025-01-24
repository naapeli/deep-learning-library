import torch
from collections import deque
import random
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from src.DLL.DeepLearning.Model import Model
from src.DLL.DeepLearning.Layers import Dense
from src.DLL.DeepLearning.Layers.Activations import ReLU
from src.DLL.DeepLearning.Optimisers import ADADELTA
from src.DLL.DeepLearning.Initialisers import Kaiming_Uniform


GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 32
MEMORY_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 5000
TARGET_UPDATE_FREQ = 10
EPISODES = 6000

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, truncated = zip(*batch)
        return (
            torch.stack([torch.tensor(state, dtype=torch.float32) for state in states]),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack([torch.tensor(state, dtype=torch.float32) for state in next_states]),
            torch.tensor(dones, dtype=torch.float32),
            torch.tensor(truncated, dtype=torch.float32)
        )

    def size(self):
        return len(self.buffer)

def epsilon_greedy_policy(state, q_network, epsilon, action_space):
    if random.random() < epsilon:
        return action_space.sample()
    else:
        state = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0)
        return q_network.predict(state).argmax(dim=1).item()

env = gym.make("CartPole-v1")#, render_mode="human"
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

hidden_dim = 16
q_network = Model(int(state_dim))
q_network.add(Dense(hidden_dim, activation=ReLU(), initialiser=Kaiming_Uniform()))
q_network.add(Dense(hidden_dim, activation=ReLU(), initialiser=Kaiming_Uniform()))
q_network.add(Dense(hidden_dim, activation=ReLU(), initialiser=Kaiming_Uniform()))
q_network.add(Dense(int(action_dim), initialiser=Kaiming_Uniform()))
q_network.compile(optimiser=ADADELTA(LR))  # default to MSE loss
target_network = q_network.clone()
replay_buffer = ReplayBuffer(MEMORY_SIZE)

epsilon = EPSILON_START
epsilon_decay = (EPSILON_START - EPSILON_END) / EPSILON_DECAY

rewards_memory = []

for episode in range(EPISODES):
    state = env.reset()[0]
    total_reward = 0

    while True:
        action = epsilon_greedy_policy(state, q_network, epsilon, env.action_space)
        next_state, reward, done, truncated, info = env.step(action)

        replay_buffer.add((state, action, reward, next_state, done, truncated))

        state = next_state
        total_reward += reward

        if replay_buffer.size() >= BATCH_SIZE:
            states, actions, rewards, next_states, dones, truncateds = replay_buffer.sample(BATCH_SIZE)

            # Compute target Q-values
            next_q_values = target_network.predict(next_states).max(dim=1)[0]
            targets = rewards + GAMMA * next_q_values * (1 - dones)

            # Compute current Q-values
            # states.requires_grad = True
            # states.retain_grad()
            q_values = q_network.predict(states, training=True)
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            initial_gradient = q_network.loss.gradient(q_values, targets)
            # q_network.loss.loss(q_values, targets).backward()
            gradient = torch.zeros((BATCH_SIZE, action_dim))
            gradient[torch.arange(BATCH_SIZE), actions] = initial_gradient  # backpropagate through the q_values.gather line
            # print(states.grad == q_network.backward(gradient, training=True))  # gradients are correct with autograd and my implementation
            q_network.backward(gradient, training=True)
            q_network.optimiser.update_parameters()

        if done or truncated:
            break

    epsilon = max(EPSILON_END, epsilon - epsilon_decay)

    if episode % TARGET_UPDATE_FREQ == 0:
        target_network = q_network.clone()

    rewards_memory.append(total_reward)
    print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

# plot how the reward changed during training
plt.plot(rewards_memory)
plt.ylabel("Total reward")
plt.xlabel("Episode")
plt.show()


# render an example cart
env = gym.make("CartPole-v1", render_mode="human")
state = env.reset()[0]
total_reward = 0
steps = []
done = False

stop = ""
while stop != "stop":
    # Run one episode
    state = env.reset()[0]
    done = False
    env.render()
    while not done:
        state = torch.from_numpy(state).unsqueeze(0)
        best_action = target_network.predict(state).argmax(dim=1).item()
        
        next_state, reward, done, truncated, info = env.step(best_action)
        total_reward += reward
        state = next_state
    stop = input('Write "stop" to stop animation: ')

env.close()
