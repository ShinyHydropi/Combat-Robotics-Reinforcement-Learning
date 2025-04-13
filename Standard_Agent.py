from __future__ import annotations

from collections import defaultdict

import numpy as np
from tqdm import tqdm
import time
import gymnasium as gym
import arena
import math
import matplotlib.pyplot as plt
import pickle

adv_type = ""
while not adv_type in ["aggressive", "defensive", "mixed", "human"]:
    adv_type = input("Adversary policy (aggressive, defensive, mixed, human): ")
adversary = ["aggressive", "defensive", "mixed", "human"].index(adv_type)
env = gym.make("arena", render_mode="human" if adversary == 3 else None, size = 1, adversary = adversary)


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
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.counts = defaultdict(lambda: np.zeros(env.action_space.n))
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.performance = []
        
# outputs an action given the current observation
    def get_action(self, env, obs: dict[str: list[int,int,int]]) -> int:
        if np.random.random() < self.epsilon:
            return env.action_space.sample()

# with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[(obs["agent"][0], obs["agent"][1], obs["agent"][2], obs["adversary"][0], obs["adversary"][1], obs["adversary"][2])]))

# updates Q-values at the end of an episode
    def update(
        self, movelog: list[tuple[int,int,int,int,int,int], int], reward: float
    ):
        reward_total = reward
        self.performance.append(reward + (0 if len(self.performance) == 0 else self.performance[-1]))
        for move in movelog:
            self.q_values[move[0]][move[1]] = (
                (self.q_values[move[0]][move[1]] * self.counts[move[0]][move[1]] + reward_total) / (self.counts[move[0]][move[1]] + 1)
            )
            self.counts[move[0]][move[1]] += 1
            reward_total = self.discount_factor * reward_total


    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

n_episodes = 0
while n_episodes < 10:
    try:
        n_episodes = int(input("Number of episodes (integer greater than 9): "))
    except:
        pass
epsilon_decay = 2/n_episodes
final_epsilon = 0.1
start_epsilon = 1

agent = GridAgent(
    env=env,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

if __name__ == '__main__':
    tot_epi_len = 0
    # simulates episodes
    for episode in tqdm(range(n_episodes)):
    
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
    
    x = range(n_episodes)
    y = agent.performance
    fig, ax = plt.subplots()

    ax.plot(x, y, linewidth=2.0)

    ax.set(xlim=(0, n_episodes), xticks = range(0, n_episodes, n_episodes//10), ylim = (-n_episodes, n_episodes))
    plt.savefig("Performance.png")
