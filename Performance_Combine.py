import csv
import numpy as np
from tqdm import tqdm
import gymnasium as gym
import arena
import pickle

performances = [[],[],[]]
for i in range(1,4):
    for j in range(1,29):
        with open("/home/freddy/AICRL/Agent_" + str(8*i) + "x" + str(8*i) + "/Session" + str(min(j, 15)) + ".pkl", "rb") as read_file:
            temp = performances[i - 1][len(performances[i - 1]) - 1] if j > 1 else 0
            contents = pickle.load(read_file)
            for item in range(99999, 1000000, 100000):
                performances[i - 1].append(contents[item] + temp)
data = [list(row) for row in zip(*performances)]
data.insert(0,["8 x 8", "16 x 16", "24 x 24"])
with open("/home/freddy/AICRL/performances.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)
