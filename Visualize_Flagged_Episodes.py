import csv #imports
import numpy as np
from tqdm import tqdm
import time
import gymnasium as gym
import arena
import pickle
    

env = gym.make("arena", render_mode="human", adversary = 1, fps = 4)#makes arena environment with human visibility, defensive adversary, and 4 fps

action_to_direction = {#list of action-direction data
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

with open("Flagged_Episodes.pkl", "rb") as read_file:
    seed, flags, control = pickle.load(read_file)#opens saved seed, flagged episodes, and control policy

def translate(robot, action, path_steps = 1):#moves the object based on the action selected
    direction = action_to_direction[action] * np.array([1 / path_steps, 1, 1])
    return np.clip(np.array([robot[0] + direction[1] / path_steps * np.cos(robot[2] + np.pi/2), robot[1] - direction[1] / path_steps * np.sin(robot[2] + np.pi/2), robot[2]]) if direction[0] == 0 else np.array([robot[0] + direction[1] * (np.cos(direction[0] + robot[2] + direction[2]) - np.cos(robot[2] + direction[2])), robot[1] + direction[1] * -(np.sin(direction[0] + robot[2] + direction[2]) - np.sin(robot[2] + direction[2])), (robot[2] + direction[0])%(2 * np.pi)]), [6.45, 6.45, 0], [89.55, 89.55, 2 * np.pi])
    
def disk(robot, scale = 1):#spinner center
    return (scale * (robot[0] - 3.2 * np.sin(robot[2])), scale * (robot[1] - 3.2 * np.cos(robot[2])))
    
def line_point(x1: float, y1: float, x2: float, y2: float, xp: float, yp: float):#line intersecting point
    return 0.1 > abs(np.hypot(x1 - xp, y1 - yp) + np.hypot(x2 - xp, y2 - yp) - np.hypot(x1 - x2, y1 - y2))
    
    
def circle_point(xc: float, yc: float, r: float, xp: float, yp: float):#circle intersecting point
    return r >= np.hypot(xp - xc, yp - yc)
    
    
def line_circle(x1: float, y1: float, x2: float, y2: float, xc: float, yc: float, r: float):#
    if (circle_point(xc, yc, r, x1, y1) or circle_point(xc, yc, r, x2, y2)):
        return True
    dot = (((xc - x1) * (x2 - x1)) + ((yc - y1) * (y2 - y1))) / ((x1 - x2)**2 + (y1 - y2)**2)
    xnear = x1 + dot * (x2 - x1)
    ynear = y1 + dot * (y2 - y1)
    return circle_point(xc, yc, r, xnear, ynear) and line_point(x1, y1, x2, y2, xnear, ynear)

#Collect policies
policies = []
for i in range(1,3):
    h,k = 8*i,8
    with open("/home/freddy/AICRL/Agent_" + str(8*i) + "x" + str(8*i) + "/Current.pkl", "rb") as read_file:
        save_data = pickle.load(read_file)[1]
    policies.append({(a,b,c,d,e,f):save_data[f + k*e + k*h*d + k*h*h*c + k*k*h*h*b + k*k*h**3*a] for a in tqdm(range(h)) for b in range(h) for c in range(k) for d in range(h) for e in range(h) for f in range(k)})
save_values = ()
for i in tqdm(range(27)):
    with open("/home/freddy/AICRL/Agent_24x24/Current-" + str(i) + ".pkl", "rb") as read_file:
        temp_values = pickle.load(read_file)[1]
    save_values += temp_values
h,k = 24,8
policies.append({(a,b,c,d,e,f):save_values[f + k*e + k*h*d + k*h*h*c + k*k*h*h*b + k*k*h**3*a] for a in tqdm(range(h)) for b in range(h) for c in range(k) for d in range(h) for e in range(h) for f in range(k)})


#Aggressive policy action selection
def Aggressive_select(pos):
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


def defensive_select(pos):
    agent_location = pos["agent"]
    adversary_location = pos["adversary"]
    test = translate(agent_location, 8)
    if (agent_location != test).any() and line_circle(agent_location[0], agent_location[1], test[0], test[1], adversary_location[0], adversary_location[1], 3.9105):
        test = translate(agent_location, 8, 20)
    temp_angle = np.absolute(2*np.pi - (test[2]+np.pi/2)%(2*np.pi) - np.arctan2(adversary_location[1] - test[1], adversary_location[0] - test[0])%(2*np.pi))
    angle = min(temp_angle, 2*np.pi - temp_angle)
    action = 8
    for index in range(len(action_to_direction) - 1):
        test = translate(agent_location, index)
        temp_angle = np.absolute(2*np.pi - (test[2]+np.pi/2)%(2*np.pi) - np.arctan2(adversary_location[1] - test[1], adversary_location[0] - test[0])%(2*np.pi))
        if (min(temp_angle, 2*np.pi - temp_angle) < angle):
            angle = min(temp_angle, 2*np.pi - temp_angle)
            action = index            
    return action


def set_episode(seed, n, policy, agent):
    env = gym.make("arena", render_mode="human", adversary = 1, size = agent, fps = 4)
    obs, info = env.reset(seed = seed, options = policy)
    for i in range(n):
        obs, info = env.reset(options = policy)
    env.render()
    return obs, info, env

if __name__ == '__main__':
    indexes = np.random.choice(range(len(flags)), 20, False)
    for i in tqdm(indexes):
        episode, agent = flags[i]
        env.close()
        obs, info, env = set_episode(seed, episode, control, agent)
        time.sleep(2)
        done = False
        while not done:
            if control == "aggressive":
                action = aggressive_select(info)
            elif control == "defensive":
                action = defensive_select(info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
        env.close()
        obs, info, env = set_episode(seed, episode, "learned", agent)
        time.sleep(2)
        done = False
        while not done:
            action = int(np.argmax(policies[agent - 1][(obs["agent"][0], obs["agent"][1], obs["agent"][2], obs["adversary"][0], obs["adversary"][1], obs["adversary"][2])]))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
    env.close()
