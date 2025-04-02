from __future__ import annotations

import numpy as np
from tqdm import tqdm
import time
import gymnasium as gym
import arena
import matplotlib.pyplot as plt
import pickle
import math

start = time.time()
size = 1
file_path = "/home/freddy/AICRL/Agent_" + str(8*size) + "x" + str(8*size)
env = gym.make("arena", render_mode=None, size = size, adversary = 1)
with open(file_path + "/Current.pkl", "rb") as read_file:
    try: # attempt to read previously saved Q-table and counts
        save_data = pickle.load(read_file)
        print(save_data[0] + 1)
    except:
        save_data = None
        print(1)

# Agent class

class GridAgent:
    def __init__(
        self,
        env,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        h = env.observation_space["agent"].high[0]+1
        k = env.observation_space["agent"].high[2]+1
        if save_data != None: # loads a Q-table if one is saved
            self.q_values = {(a,b,c,d,e,f):save_data[1][f + k*e + k*h*d + k*h*h*c + k*k*h*h*b + k*k*h**3*a] for a in tqdm(range(h)) for b in range(h) for c in range(k) for d in range(h) for e in range(h) for f in range(k)}
            self.counts = {(a,b,c,d,e,f):save_data[2][f + k*e + k*h*d + k*h*h*c + k*k*h*h*b + k*k*h**3*a] for a in tqdm(range(h)) for b in range(h) for c in range(k) for d in range(h) for e in range(h) for f in range(k)}
        else: # creates a blank Q-table if none are saved
            self.q_values = {(a,b,c,d,e,f):np.zeros(env.action_space.n) for a in tqdm(range(h)) for b in range(h) for c in range(k) for d in range(h) for e in range(h) for f in range(k)}
            self.counts = {(a,b,c,d,e,f):np.zeros(env.action_space.n) for a in tqdm(range(h)) for b in range(h) for c in range(k) for d in range(h) for e in range(h) for f in range(k)}
        
        
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.performance = []
        
    # outputs an action given the current observation
    def get_action(self, env, obs: dict[str: list[int,int,int]]) -> int:
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        else: # with probability (1 - epsilon) act greedily (exploit)
            return int(np.argmax(self.q_values[(obs["agent"][0], obs["agent"][1], obs["agent"][2], obs["adversary"][0], obs["adversary"][1], obs["adversary"][2])]))

    # updates Q-values at the end of an episode
    def update(
        self, movelog: list[tuple[int,int,int,int,int,int], int], reward: float
    ):
        reward_total = reward
        self.performance.append(reward + (0 if len(self.performance) == 0 else self.performance[len(self.performance)-1]))
        for move in movelog: # the Q-value for each state-action pair is updated according to Monte Carlo methods
            self.q_values[move[0]][move[1]] = (
                (self.q_values[move[0]][move[1]] * self.counts[move[0]][move[1]] + reward_total) / (self.counts[move[0]][move[1]] + 1)
            )
            self.counts[move[0]][move[1]] += 1
            reward_total = self.discount_factor * reward_total


    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


n_episodes = 1000000 # parameters for learning set for learning 1,000,000 episode at a time for 28 sessions
epsilon_decay = 1/14000000
final_epsilon = 0.1
start_epsilon = max(final_epsilon, 1 - (save_data[0] if save_data != None else 0) * n_episodes * epsilon_decay)

agent = GridAgent(
    env=env,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

if __name__ == '__main__':
    n = save_data[0] if save_data != None else 0
    tot_epi_len = 0
    for episode in tqdm(range(n_episodes)): # simulates episodes
        obs = env.reset()[0]
        done = False
        movelog = []
        while not done:
            action = agent.get_action(env, obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            movelog.insert(0, ((obs["agent"][0], obs["agent"][1], obs["agent"][2], obs["adversary"][0], obs["adversary"][1], obs["adversary"][2]), action))
            done = terminated or truncated
            obs = next_obs
        tot_epi_len += info["time"]
        agent.update(movelog, reward)
        agent.decay_epsilon()
            
    

    print("State space (independent): 2^" + str(2 * math.log((env.observation_space["agent"].high[0] + 1) * (env.observation_space["agent"].high[1] + 1) * (env.observation_space["agent"].high[2] + 1), 2)))
    print("Average episode length (dependent, time-steps): " + str(tot_epi_len/n_episodes))
    print("Average visits per state (dependent): " + str(tot_epi_len/((env.observation_space["agent"].high[1] + 1)**4)))
    
    x = range(n_episodes)
    y = agent.performance
    fig, ax = plt.subplots()

    ax.plot(x, y, linewidth=2.0)

    ax.set(xlim=(0, n_episodes), xticks = range(0, n_episodes, 100000), ylim=(-500000, 500000))
    n += 1
    plt.savefig(file_path + "/Performance" + str(n) +".png") # saves a graph of the agents cumulative performance in this training session
    print("done file 1")
    with open(file_path + "/Current.pkl", 'wb') as created_file:
        pickle.dump((n, tuple(agent.q_values.values()), tuple(agent.counts.values())), created_file) # saves the current session number and network object
    print("done file 2")
    with open(file_path + "/Session" + str(n) + ".pkl", 'wb') as created_file:
        pickle.dump(agent.performance, created_file) # saves the cumulative performance from this session of training
    print("done file 3")
    end = time.time()
    print((end-start)/3600) # prints time elapsed in hours
