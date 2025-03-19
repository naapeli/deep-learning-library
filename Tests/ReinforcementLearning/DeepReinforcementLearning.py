import torch
from collections import deque
import random
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from src.DLL.DeepLearning.Model import Model
from src.DLL.DeepLearning.Layers import Dense
from src.DLL.DeepLearning.Layers.Activations import ReLU
from src.DLL.DeepLearning.Optimisers import RMSPROP, ADAM
from src.DLL.DeepLearning.Initialisers import Kaiming_Uniform


GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 128
MEMORY_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPISODES = 5000  # For optimal results, one should experiment with increasing the amount of episodes in training. About 5000 to 10000 episodes should be sufficient for pretty much optimal agent.
MOVING_AVERAGE = 50
MAX_EPISODE_LENGTH = 500

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack([torch.tensor(state, dtype=torch.float32) for state in states]),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack([torch.tensor(state, dtype=torch.float32) for state in next_states]),
            torch.tensor(dones, dtype=torch.float32)
        )

    def size(self):
        return len(self.buffer)

def epsilon_greedy_policy(state, agent, epsilon, action_space):
    if random.random() < epsilon:
        return action_space.sample()
    else:
        state = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0)
        return agent.predict(state).argmax(dim=1).item()

env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

hidden_dim = 24
agent = Model(int(state_dim))
agent.add(Dense(hidden_dim, activation=ReLU(), initialiser=Kaiming_Uniform()))
agent.add(Dense(hidden_dim, activation=ReLU(), initialiser=Kaiming_Uniform()))
agent.add(Dense(int(action_dim), initialiser=Kaiming_Uniform()))
agent.compile(optimiser=RMSPROP(learning_rate=LR))  # default to MSE loss, The chosen optimiser affects the training results VERY much. ADAM and RMSPROP produce good results, while other optimisers fail to reach the optimum.
replay_buffer = ReplayBuffer(MEMORY_SIZE)

epsilon = EPSILON_START
epsilon_decay = (EPSILON_START - EPSILON_END) / EPISODES

rewards_memory = []

for episode in range(EPISODES):
    state = env.reset()[0]
    total_reward = 0

    step = 0
    while True:
        step += 1
        action = epsilon_greedy_policy(state, agent, epsilon, env.action_space)
        next_state, reward, done, truncated, info = env.step(action)

        old_state = state
        state = next_state
        total_reward += reward

        if done or truncated:
            if step < MAX_EPISODE_LENGTH:
                reward = -5  # make the reward lower if the cart fails to balance or goes out-of-bounds.
            if step == MAX_EPISODE_LENGTH:
                reward = 5
            replay_buffer.add((old_state, action, reward, next_state, done or truncated))
            break
        
        # Modify the reward to try to keep the pole in the center. These lines can be removed, however, they make the training faster and more stable.
        # If these lines are included, the agent does not fail, whereas, if they are not, the agent might make a mistake. The istake almost always
        # after collecting 500 total reward, which was considerd to be good enough for the agent to stop the training.

        # Locations
        reward -= abs(state[0]) / 4.8
        reward -= abs(state[2]) / 0.418

        # Speeds
        # reward -= abs(state[1]) / 5
        # reward -= abs(state[3]) / 5

        replay_buffer.add((old_state, action, reward, next_state, done or truncated))

    epsilon = max(EPSILON_END, epsilon - epsilon_decay)

    # Train the model
    if replay_buffer.size() >= BATCH_SIZE:
        states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

        agent.optimiser.zero_grad()

        # Compute target Q-values
        next_q_values = agent.predict(next_states).max(dim=1)[0]
        targets = rewards + GAMMA * next_q_values * (1 - dones)

        # Compute current Q-values
        q_values = agent.predict(states, training=True)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        initial_gradient = agent.loss.gradient(q_values, targets)
        gradient = torch.zeros((BATCH_SIZE, action_dim))
        gradient[torch.arange(BATCH_SIZE), actions] = initial_gradient  # backpropagate through the q_values.gather line
        agent.backward(gradient, training=True)
        agent.optimiser.update_parameters()

    rewards_memory.append(total_reward)
    print(f"Episode {episode + 1} | Total Reward: {int(total_reward)} / 500 | Epsilon: {epsilon:.2f}")

    if sum(rewards_memory[-MOVING_AVERAGE:]) / MOVING_AVERAGE > 490:
        print("Average reward is over 490. The training is stopped.")
        break

# plot how the reward changed during training
plt.plot(rewards_memory)
plt.plot(np.arange(MOVING_AVERAGE - 1, len(rewards_memory)), np.convolve(rewards_memory, np.ones(MOVING_AVERAGE) / MOVING_AVERAGE, mode='valid'))
plt.ylabel("Total reward")
plt.xlabel("Episode")
plt.show()


# render an example cart
env = gym.make("CartPole-v1", render_mode="human")
state = env.reset()[0]
steps = []

stop = ""
while stop != "stop":
    state = env.reset()[0]
    done = False
    total_reward = 0
    env.render()
    while not done:
        state = torch.from_numpy(state).unsqueeze(0)
        best_action = agent.predict(state).argmax(dim=1).item()
        
        next_state, reward, done, truncated, info = env.step(best_action)
        total_reward += reward
        state = next_state
    
    print(f"Total reward: {total_reward}")
    stop = input('Write "stop" to stop animation or press "enter" to play another episode: ')

env.close()
