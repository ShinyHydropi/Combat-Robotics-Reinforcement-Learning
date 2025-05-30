from __future__ import annotations
import numpy as np
from tqdm import tqdm
import gymnasium as gym
import arena
import matplotlib.pyplot as plt
import pickle
import math

adv_type = ""
while not adv_type in ["aggressive", "defensive", "mixed", "human"]:
    adv_type = input("Adversary policy (aggressive, defensive, mixed, human): ")
adversary = ["aggressive", "defensive", "mixed", "human"].index(adv_type)
env = gym.make("arena", render_mode="human" if adversary == 3 else None, size = 1, adversary = adversary)


# Network Class

class Network:
    def __init__(self, layer_shape: list, activation_function: str):
        self.activation = activation_function
        self.weights = []
        self.shape = layer_shape
        for i in range(len(self.shape) - 1): # initialize weights with a normal Xavier distribution
            limit = np.sqrt(2/(self.shape[i]+ self.shape[i + 1]))
            self.weights.append(np.random.normal(scale=limit, size=(self.shape[i + 1], self.shape[i])))
    
    # outputs the result of matrix multiplication of the layers with the input
    def feed(self, input_activation):
        current_activation = input_activation.transpose()
        for layer in self.weights:
            product = np.dot(layer, current_activation)
            if self.activation == "Sigmoid":
                current_activation = 1/(1 + np.exp(-product))
            if self.activation == "Tanh":
                current_activation = np.tanh(product)
        return current_activation
    
    # compute the gradient for a given state
    def gradient(self, input_activation):
        self.means = [] # mean and variance is collected for evaluation of the network's performance
        self.vars = []
        activations = [input_activation.transpose()]
        for layer in self.weights: # a list of activations is saved for computing partial derivatives
            product = np.dot(layer, activations[-1])
            str(product)
            if self.activation == "Sigmoid":
                activations.append(1/(1 + np.exp(-product)))
            if self.activation == "Tanh":
                activations.append(np.tanh(product))
            self.means.append(np.mean(activations[-1]))
            self.vars.append(np.var(activations[-1]))
        delta_w = []
        for i in range(len(self.shape) - 1):
            delta_w.append(np.zeros((self.shape[i+1], self.shape[i])))
        if self.activation == "Sigmoid": # weights and activations are used to compute the partial derivatives of the last layer
            delta_w[-1][np.argmax(activations[-1])] = (activations[-1].max() - 1) * activations[-1].max() * activations[-2]
            delta_a = (activations[-1].max() - 1) * activations[-1].max() * self.weights[-1][np.argmax(activations[-1])]
        if self.activation == "Tanh":
            delta_w[-1][np.argmax(activations[-1])] = (1 - activations[-1].max()**2) * activations[-2]
            delta_a = (1 - activations[-1].max()**2) * self.weights[-1][np.argmax(activations[-1])]
        for i in range(-2, -len(self.shape),-1):
            temp = np.zeros((self.shape[i - 1],))
            for j in range(self.shape[i]): # back propagation is used to compute the gradient of the remaining layers
                if self.activation == "Sigmoid":
                    delta_w[i][j] = (activations[i][j] - 1) * activations[i][j] * activations[i - 1] * delta_a[j]
                    temp += (activations[i][j] - 1) * activations[i][j] * self.weights[i][j] * delta_a[j]
                if self.activation == "Tanh":
                    delta_w[i][j] = (1 - activations[i][j]**2) * activations[i - 1] * delta_a[j]
                    temp += (1 - activations[i][j]**2) * self.weights[i][j] * delta_a[j]
            delta_a = temp
        return delta_w
        




# Agent class

class NetworkAgent:
    def __init__(
        self,
        env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        activation_f: str,
        discount_factor: float = 0.95
    ):
        try: # a previously saved network is attempted to be read
            with open(activation_f + "_Network/network.pkl", "rb") as read_file:
                save_data = pickle.load(read_file)
        except:
            save_data = (0, None)
        if save_data[1] == None:
            self.network = Network(layer_shape = [6,4,4,9], activation_function = activation_f) # if there was no saved network, a new one is created
        else:
            self.network = save_data[1]
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = max(final_epsilon, initial_epsilon - (save_data[0] if save_data != None else 0) * n_episodes * epsilon_decay)
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.performance = []
        self.n = save_data[0] + 1
        print(self.n)
        
    # outputs an action given the current observation
    def get_action(self, env, obs: list[float,float,float,float,float,float]) -> int:
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        else: # with probability (1 - epsilon) act greedily (exploit)
            return int(np.argmax(self.network.feed(obs)))

    # updates Q-values at the end of an episode
    def update(
        self, movelog, reward: float
    ):
        returns = [reward]
        for move in movelog:
            returns.insert(0,self.discount_factor * returns[0]) # list of returns is calculated to determine the weight updates
        self.performance.append(reward + (0 if len(self.performance) == 0 else self.performance[len(self.performance)-1]))
        updates = []
        for i in range(len(self.network.shape) - 1):
            updates.append(np.zeros((self.network.shape[i+1], self.network.shape[i])))
        for m in range(len(movelog)):
            gradient = self.network.gradient(movelog[m][0])
            for layer in range(len(self.network.weights)):
                updates[layer] += self.learning_rate * (returns[m] - self.network.feed(movelog[m][0]).max()) * gradient[layer] # updates are computed with Monte Carlo methods
        for layer in range(len(self.network.weights)):
            self.network.weights[layer] += updates[layer]
    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay

learning_rate = 0.01
n_episodes = 0
while n_episodes < 10:
    try:
        n_episodes = int(input("Number of episodes (integer greater than 9): "))
    except:
        pass
        
epsilon_decay = 4**(-1/n_episodes)
final_epsilon = 0.1
start_epsilon = 1
activation = ""
while not activation in ["Sigmoid", "Tanh"]:
    activation = input("Activation function (Sigmoid, Tanh): ")

agent = NetworkAgent(
    env=env,
    learning_rate = learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    activation_f = activation
)


if __name__ == '__main__':
    tot_epi_len = 0
    for episode in tqdm(range(n_episodes)): # simulates episodes
    
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
    x = range(n_episodes)
    y = agent.performance
    fig, ax = plt.subplots()

    ax.plot(x, y, linewidth=2.0)

    ax.set(xlim=(0, n_episodes), xticks = range(0, n_episodes, n_episodes//10), ylim = (-n_episodes, n_episodes))
    plt.savefig(activation + "_Network/Performance" + str(agent.n) + ".png") # saves a graph of the agents cumulative performance in this training session
    print("done file 1")
    with open(activation + "_Network/network.pkl", 'wb') as created_file:
        pickle.dump((agent.n, agent.network), created_file) # saves the current session number and network object
    print("done file 2")
    with open(activation + "_Network/Session" + str(agent.n) + ".pkl", 'wb') as created_file:
        pickle.dump(agent.performance, created_file) # saves the cumulative performance from this session of training
    print("done file 3")
