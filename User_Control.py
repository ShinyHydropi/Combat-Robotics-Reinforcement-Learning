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
import keyboard


adv_type = ""
while not adv_type in ["aggressive", "defensive", "mixed", "human", "8x8", "16x16", "24x24"]:
    adv_type = input("Adversary policy (aggressive, defensive, mixed, human, 8x8, 16x16, 24x24): ")
adversary = ["aggressive", "defensive", "mixed", "human", "8x8", "16x16", "24x24"].index(adv_type)
env = gym.make("arena", render_mode="human", size = 1, adversary = 3 if adversary > 2 else adversary)

n_episodes = 0
while n_episodes < 10:
    try:
        n_episodes = int(input("Number of episodes (integer greater than 9): "))
    except:
        pass

if adversary < 4:
    print("Blue robot controlled with A,W,D")
if adversary > 2:
    print("Red robot controlled with arrow keys")
time.sleep(3)


#Collect policies
i = adversary - 3
if i in range(1,3):
    h,k = 8*i,8
    with open("/home/freddy/AICRL/Agent_" + str(8*i) + "x" + str(8*i) + "_Aggressive/Current.pkl", "rb") as read_file:
        save_data = pickle.load(read_file)[1]
    policy = {(a,b,c,d,e,f):save_data[f + k*e + k*h*d + k*h*h*c + k*k*h*h*b + k*k*h**3*a] for a in tqdm(range(h)) for b in range(h) for c in range(k) for d in range(h) for e in range(h) for f in range(k)}
if i == 3:
    save_values = ()
    for j in tqdm(range(27)):
        with open("/home/freddy/AICRL/Agent_24x24_Aggressive/Current-" + str(j) + ".pkl", "rb") as read_file:
            temp_values = pickle.load(read_file)[1]
        save_values += temp_values
    h,k = 24,8
    policy = {(a,b,c,d,e,f):save_values[f + k*e + k*h*d + k*h*h*c + k*k*h*h*b + k*k*h**3*a] for a in tqdm(range(h)) for b in range(h) for c in range(k) for d in range(h) for e in range(h) for f in range(k)}


if __name__ == '__main__':
    tot_epi_len = 0
    performance = []
    # simulates episodes
    for episode in tqdm(range(n_episodes)):
    
        obs = env.reset()[0]
        done = False
        while not done:
            if adversary < 4:
                kb = keyboard.get_hotkey_name().split("+")
                direction = []
                if "a" in kb:
                    direction.append(0)
                if "w" in kb:
                    direction.append(2)
                if "d" in kb:
                    direction.append(4)
                if len(direction) > 0:
                    action = [2,6,8,7,3][int(np.average(direction))]
                else:
                    action = None
            else:
                action = int(np.argmax(policy[(obs["agent"][0], obs["agent"][1], obs["agent"][2], obs["adversary"][0], obs["adversary"][1], obs["adversary"][2])]))
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            obs = next_obs
        performance.append(reward + (0 if len(performance) == 0 else performance[-1]))
        tot_epi_len += info["time"]
    print("Average episode length (dependent, time-steps): " + str(tot_epi_len/n_episodes))
    env.close()
    
    x = range(n_episodes)
    y = performance
    fig, ax = plt.subplots()

    ax.plot(x, y, linewidth=2.0)

    ax.set(xlim=(0, n_episodes), xticks = range(0, n_episodes, n_episodes//10), ylim = (-n_episodes, n_episodes))
    plt.savefig("Performance.png")
