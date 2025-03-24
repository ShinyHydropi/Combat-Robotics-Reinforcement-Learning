from __future__ import annotations

from collections import defaultdict

import numpy as np
from tqdm import tqdm
import time
import gymnasium as gym
import arena
import math
import multiprocessing
import matplotlib.pyplot as plt
import pickle

env = gym.make("arena", render_mode=None, size = 1, adversary = 2)

# Network Class

class Network:
    def __init__(self, layer_shape: list):
        self.weights = []
        self.shape = layer_shape
        for i in range(len(self.shape) - 1):
            limit = np.sqrt(2/(self.shape[i]+ self.shape[i + 1]))
            self.weights.append(np.random.normal(scale=limit, size=(self.shape[i + 1], self.shape[i])))
    
    def feed(self, input_activation):
        current_activation = input_activation.transpose()
        for layer in self.weights:
            product = np.dot(layer, current_activation)
            #current_activation = 1/(1 + np.exp(-product))
            current_activation = np.tanh(product)
        return current_activation
        
    def gradient(self, input_activation):
        self.means = []
        self.vars = []
        activations = [input_activation.transpose()]
        for layer in self.weights:
            product = np.dot(layer, activations[-1])
            str(product)
            #activations.append(1/(1 + np.exp(-product)))
            activations.append(np.tanh(product))
            self.means.append(np.mean(activations[-1]))
            self.vars.append(np.var(activations[-1]))
        delta_w = []
        for i in range(len(self.shape) - 1):
            delta_w.append(np.zeros((self.shape[i+1], self.shape[i])))
        #delta_w[-1][np.argmax(activations[-1])] = (activations[-1].max() - 1) * activations[-1].max() * activations[-2]
        #delta_a = (activations[-1].max() - 1) * activations[-1].max() * self.weights[-1][np.argmax(activations[-1])]
        delta_w[-1][np.argmax(activations[-1])] = (1 - activations[-1].max()**2) * activations[-2]
        delta_a = (1 - activations[-1].max()**2) * self.weights[-1][np.argmax(activations[-1])]
        for i in range(-2, -len(self.shape),-1):
            temp = np.zeros((self.shape[i - 1],))
            for j in range(self.shape[i]):
                #delta_w[i][j] = (activations[i][j] - 1) * activations[i][j] * activations[i - 1] * delta_a[j]
                #temp += (activations[i][j] - 1) * activations[i][j] * self.weights[i][j] * delta_a[j]
                delta_w[i][j] = (1 - activations[i][j]**2) * activations[i - 1] * delta_a[j]
                temp += (1 - activations[i][j]**2) * self.weights[i][j] * delta_a[j]
            delta_a = temp
        return delta_w
        

with open("Network.pkl", "rb") as read_file:
    save_data = pickle.load(read_file)
print(save_data.weights[0][0])


# Agent class

class GridAgent:
    def __init__(
        self,
        env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        self.network = Network(layer_shape = [6,32,32,9])
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.performance = []
        
# outputs an action given the current observation
    def get_action(self, env, obs: list[float,float,float,float,float,float]) -> int:
        if np.random.random() < self.epsilon:
            return env.action_space.sample()

# with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.network.feed(obs)))

# updates Q-values at the end of an episode
    def update(
        self, movelog, reward: float
    ):
        returns = [reward]
        for move in movelog:
            returns.insert(0,self.discount_factor * returns[0])
        self.performance.append(reward + (0 if len(self.performance) == 0 else self.performance[len(self.performance)-1]))
        updates = []
        for i in range(len(self.network.shape) - 1):
            updates.append(np.zeros((self.network.shape[i+1], self.network.shape[i])))
        for m in range(len(movelog)):
            gradient = self.network.gradient(movelog[m][0])
            for layer in range(len(self.network.weights)):
                updates[layer] += self.learning_rate * (returns[m] - self.network.feed(movelog[m][0]).max()) * gradient[layer]
        for layer in range(len(self.network.weights)):
            self.network.weights[layer] += updates[layer]


    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

learning_rate = 0.01
n_episodes = 100
epsilon_decay = 2/n_episodes
final_epsilon = 0.1
start_epsilon = 1

agent = GridAgent(
    env=env,
    learning_rate = learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)
print()


if __name__ == '__main__':
    tot_epi_len = 0
    # simulates episodes
    for episode in tqdm(range(n_episodes)):
    
        info = env.reset()[1]
        done = False
        movelog = []
        while not done:
            action = agent.get_action(env, info["NN"])
            next_obs, reward, terminated, truncated, info = env.step(action)
            movelog.insert(0, (info["NN"], action))
            done = terminated or truncated
            obs = next_obs
        tot_epi_len += info["time"]
        agent.update(movelog, reward)
        agent.decay_epsilon()
            
    

    print("State space (independent): 2^" + str(2 * math.log((env.observation_space["agent"].high[0] + 1) * (env.observation_space["agent"].high[1] + 1) * (env.observation_space["agent"].high[2] + 1), 2)))
    print("Average episode length (dependent, time-steps): " + str(tot_epi_len/n_episodes))
    print(agent.network.feed(env.reset()[1]["NN"]))
    print(np.mean(agent.network.means))
    print(agent.network.vars)
    with open("Network.pkl", 'wb') as created_file:
        pickle.dump(agent.network, created_file)
    print(agent.network.weights[0][0])
    x = range(n_episodes)
    y = agent.performance
    fig, ax = plt.subplots()

    ax.plot(x, y, linewidth=2.0)

    ax.set(xlim=(0, n_episodes), xticks = range(0, n_episodes, n_episodes//10), ylim = (-n_episodes//2, n_episodes//2))
    plt.savefig("Performance.png")
