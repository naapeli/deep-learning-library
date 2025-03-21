"""
Deep Q-Learning Agent for CartPole-v1
===========================================

This script implements a Deep Q-Network (DQN) by training an agent to 
balance a pole in the CartPole-v1 environment from OpenAI Gymnasium. The script also 
implements a custom training loop of a `DLL.DeepLearning.Model.Model` to train the model.

.. image:: _static/cartpole.gif

"""

import torch
from collections import deque
import random
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from DLL.DeepLearning.Model import Model
from DLL.DeepLearning.Layers import Dense
from DLL.DeepLearning.Layers.Activations import Tanh
from DLL.DeepLearning.Optimisers import RMSPROP, ADAM
from DLL.DeepLearning.Initialisers import Xavier_Normal


GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 16
MEMORY_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPISODES = 5000  # For optimal results, one should experiment with increasing the amount of episodes in training. About 5000 to 10000 episodes should be sufficient for pretty much optimal agent.
MOVING_AVERAGE = 50
MAX_EPISODE_LENGTH = 500
SAVE_IMG = False

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

torch.manual_seed(1)

hidden_dim = 64
agent = Model(int(state_dim))
agent.add(Dense(hidden_dim, activation=Tanh(), initialiser=Xavier_Normal()))
agent.add(Dense(hidden_dim, activation=Tanh(), initialiser=Xavier_Normal()))
agent.add(Dense(int(action_dim), initialiser=Xavier_Normal()))
agent.compile(optimiser=RMSPROP(learning_rate=LR))  # default to MSE loss, The chosen optimiser affects the training results VERY much. ADAM and RMSPROP produce good results, while other optimisers fail to reach the optimum.
replay_buffer = ReplayBuffer(MEMORY_SIZE)

epsilon = EPSILON_START
epsilon_decay = (EPSILON_START - EPSILON_END) / EPISODES

rewards_memory = []

for episode in range(EPISODES):
    state = env.reset(seed=episode)[0]
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
        # If these lines are included, the agent does not fail, whereas, if they are not, the agent might make a mistake. The mistake almost always
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


if SAVE_IMG:
    # render an example cart
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    while input('Write "stop" to stop the animation and press enter to rerun: ') != "stop":
        state = env.reset()[0]
        frames = []

        # Collect frames for animation
        done = False
        i = 0
        while not done:
            frames.append(env.render())  # Capture frame
            state = torch.from_numpy(state).unsqueeze(0)
            best_action = agent.predict(state).argmax(dim=1).item()
            state, _, done, _, _ = env.step(best_action)
            if i > 2000:
                break

        env.close()

        # Create animation
        fig, ax = plt.subplots()
        img = ax.imshow(frames[0])

        def update(frame):
            img.set_array(frames[frame])
            return img,

        ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=0.1, repeat=False)
        ani.save("./Docs/source/auto_examples/ReinforcementLearning/_static/cartpole.gif", writer="pillow")

        plt.show()
