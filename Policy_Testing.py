import csv
import numpy as np
from tqdm import tqdm
import gymnasium as gym
import arena
import pickle
    

env = gym.make("arena", render_mode=None)

action_to_direction = {
    0: np.array([3.334605707, 0, 0]),
    1: np.array([-3.334605707, 0, np.pi]),
    2: np.array([2.246407457, 1.139676331, 0]),
    3: np.array([-2.246407457, 1.139676331, np.pi]),
    4: np.array([1.652621231, 3.738130409, 0]),
    5: np.array([-1.652621231, 3.738130409, np.pi]),
    6: np.array([1.293401389, 6.588797033, 0]),
    7: np.array([-1.293401389, 6.588797033, np.pi]),
    8: np.array([0, 22.78940029, 0])
}

def translate(robot, action, path_steps = 1):
    direction = action_to_direction[action] * np.array([1 / path_steps, 1, 1])
    return np.clip(np.array([robot[0] + direction[1] / path_steps * np.cos(robot[2] + np.pi/2), robot[1] - direction[1] / path_steps * np.sin(robot[2] + np.pi/2), robot[2]]) if direction[0] == 0 else np.array([robot[0] + direction[1] * (np.cos(direction[0] + robot[2] + direction[2]) - np.cos(robot[2] + direction[2])), robot[1] + direction[1] * -(np.sin(direction[0] + robot[2] + direction[2]) - np.sin(robot[2] + direction[2])), (robot[2] + direction[0])%(2 * np.pi)]), [6.45, 6.45, 0], [89.55, 89.55, 2 * np.pi])
    
def disk(robot, scale = 1):
    return (scale * (robot[0] - 3.2 * np.sin(robot[2])), scale * (robot[1] - 3.2 * np.cos(robot[2])))

#Collect policies
policies = []
for i in range(1,3):
    h,k = 8*i,8
    with open("/home/freddy/AICRL/Agent_" + str(8*i) + "x" + str(8*i) + "_Aggressive/Current.pkl", "rb") as read_file:
        save_data = pickle.load(read_file)[1]
    policies.append({(a,b,c,d,e,f):save_data[f + k*e + k*h*d + k*h*h*c + k*k*h*h*b + k*k*h**3*a] for a in tqdm(range(h)) for b in range(h) for c in range(k) for d in range(h) for e in range(h) for f in range(k)})
save_values = ()
for i in tqdm(range(27)):
    with open("/home/freddy/AICRL/Agent_24x24_Aggressive/Current-" + str(i) + ".pkl", "rb") as read_file:
        temp_values = pickle.load(read_file)[1]
    save_values += temp_values
h,k = 24,8
policies.append({(a,b,c,d,e,f):save_values[f + k*e + k*h*d + k*h*h*c + k*k*h*h*b + k*k*h**3*a] for a in tqdm(range(h)) for b in range(h) for c in range(k) for d in range(h) for e in range(h) for f in range(k)})


#Chasing policy action selection
def chasing_select(pos):
    agent_location = pos["agent"]
    adversary_location = pos["adversary"]
    dist = 2000
    action = -1
    for index in range(9):
        test = translate(agent_location, index)
        temp_dist = np.hypot(adversary_location[0] - disk(test)[0], adversary_location[1] - disk(test)[1])
        if (temp_dist < dist):
            dist = temp_dist
            action = index
    return action

if __name__ == '__main__':
    #Collect chasing policy performance
    performance = [[],[],[],[]]
    infos = [[],[],[],[]]
    info = env.reset()[1]
    print(info)
    seed = env.np_random_seed
    for episode in tqdm(range(1000)):
        infos[0].append(info["agent"][0])
        done = False
        while not done:
            action = chasing_select(info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        performance[0].append(reward)
        info = env.reset()[1]
    
    #Collect learned policy performance
    for agent in range(3):
        env.close()
        env = gym.make("arena", render_mode=None, size = agent + 1)
        current_policy = policies[agent]
        obs, info = env.reset(seed = seed)
        print(info)
        for episode in tqdm(range(1000)):#000)):
            infos[agent + 1].append(info["agent"][0])
            done = False
            while not done:
                action = int(np.argmax(current_policy[(obs["agent"][0], obs["agent"][1], obs["agent"][2], obs["adversary"][0], obs["adversary"][1], obs["adversary"][2])]))
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            performance[agent + 1].append(reward)
            obs, info = env.reset()
            
    #Transpose the results
    data1 = [list(row) for row in zip(*performance)]
    data2 = [list(row) for row in zip(*infos)]
    data1.insert(0,["Aggressive", "8x8", "16x16", "24x24"])
    data2.insert(0,["Aggressive", "8x8", "16x16", "24x24"])
    #Save results
    with open("/home/freddy/AICRL/results.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data1)
    with open("/home/freddy/AICRL/infos.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data2)
