from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import time
import functools
import random
import keyboard

class Actions(Enum):
    noT_fullSL = 0
    noT_fullSR = 1
    halfT_fullSL = 2
    halfT_fullSR = 3
    fullT_fullSL = 4
    fullT_fullSR = 5
    fullT_halfSL = 6
    fullT_halfSR = 7
    fullT_noS = 8


class ArenaEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}
    
    def __init__(self, render_mode = None, size = 1, adversary = 0, fps = 100):
        self.HYPOT = np.hypot(3.9105, 4.625)
        self.RADS = np.arctan2(3.9105, 4.625)
        self.window_size = 1000  # The size of the PyGame window
        self._limit = 900
        self.size = size
        self.adversary_type = adversary
        self.current_adversary = adversary
        self.fps = fps

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(np.array([0,0,0]), np.array([8 * size - 1, 8 * size - 1, 7]), dtype = int),
                "adversary": spaces.Box(np.array([0,0,0]), np.array([8 * size - 1, 8 * size - 1, 7]), dtype = int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(9)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            Actions.noT_fullSL.value: np.array([3.334605707, 0, 0]),
            Actions.noT_fullSR.value: np.array([-3.334605707, 0, np.pi]),
            Actions.halfT_fullSL.value: np.array([2.246407457, 1.139676331, 0]),
            Actions.halfT_fullSR.value: np.array([-2.246407457, 1.139676331, np.pi]),
            Actions.fullT_fullSL.value: np.array([1.652621231, 3.738130409, 0]),
            Actions.fullT_fullSR.value: np.array([-1.652621231, 3.738130409, np.pi]),
            Actions.fullT_halfSL.value: np.array([1.293401389, 6.588797033, 0]),
            Actions.fullT_halfSR.value: np.array([-1.293401389, 6.588797033, np.pi]),
            Actions.fullT_noS.value: np.array([0, 22.78940029, 0])
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
    
        return {"agent": np.array([self._agent_location[0] // (12/self.size), self._agent_location[1] // (12/self.size), self._agent_location[2] // (np.pi/4)], dtype = "int64"), "adversary": np.array([self._adversary_location[0] // (12/self.size), self._adversary_location[1] // (12/self.size), self._adversary_location[2] // (np.pi/4)], dtype = "int64")}

    def _get_info(self):
        return {
        "time": self._time_steps,"agent": self._agent_location, "adversary": self._adversary_location, "NN": np.array([self._agent_location[0] / 96, self._agent_location[1] / 96, self._agent_location[2] / (2*np.pi), self._adversary_location[0] / 96, self._adversary_location[1] / 96, self._adversary_location[2] / (2*np.pi)])
        }

    def reset(self, seed=None, options=""):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._time_steps = 0
        # Choose the agent's location uniformly at random
        self._agent_location = np.array([self.np_random.uniform(6.45, 89.55), self.np_random.uniform(6.45, 89.55), self.np_random.uniform(0, (2 * np.pi))])

        # We will sample the target's location randomly until it does not
        # coincide with the agent's location
        self._adversary_location = self._agent_location
        while np.hypot(self._adversary_location[0] - self._agent_location[0], self._adversary_location[1] - self._agent_location[1]) < 12.9:
            self._adversary_location = np.array([self.np_random.uniform(6.45, 89.55), self.np_random.uniform(6.45, 89.55), self.np_random.uniform(0, (2 * np.pi))])
            
        #if adversary is stochastic, randomize
        if self.adversary_type == 2:
            self.current_adversary = self.np_random.integers(0,2)

        if self.render_mode == "human" and options == "":
            self._render_frame()
        else:
            self.text = options

        return self._get_obs(), self._get_info()

    def step(self, action):
        if self.current_adversary == 0:
            adv_act = self.aggressive_select()
        elif self.current_adversary == 1:
            adv_act = self.defensive_select()
        elif self.current_adversary == 3:
            kb = keyboard.get_hotkey_name().split("+")
            direction = []
            if "left" in kb:
                direction.append(0)
            if "up" in kb:
                direction.append(2)
            if "right" in kb:
                direction.append(4)
            if len(direction) > 0:
                adv_act = [2,6,8,7,3][int(np.average(direction))]
            else:
                adv_act = None
                
        for _ in range(20):
            if adv_act != None:
                self._adversary_location = self.translate(self._adversary_location, adv_act, 20)
            if action != None:
                self._agent_location = self.translate(self._agent_location, action, 20)
            if self.render_mode == "human":
                self._render_frame(self.text)
            reward, terminated = self.collision_check()
            if terminated:
                break
        self._time_steps += 1
        truncated = (self._time_steps == self._limit)
        if not terminated:
            reward = -1 if truncated else 0
        

        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    
    def corner(self, robot, n: float, scale = 1):
        if (n % 1 == 0):
            return (scale * (robot[0] + self.HYPOT * np.cos(self.RADS + robot[2] + n * np.pi)), scale * (robot[1] + self.HYPOT * -np.sin(self.RADS + robot[2] + n * np.pi)))
        else:
            return (scale * (robot[0] + self.HYPOT * np.cos(robot[2] + (n + 0.5) * np.pi - self.RADS)), scale * (robot[1] + self.HYPOT * -np.sin(robot[2] + (n + 0.5) * np.pi - self.RADS)))
    
    
    def line_point(self, x1: float, y1: float, x2: float, y2: float, xp: float, yp: float):
        return 0.1 > abs(np.hypot(x1 - xp, y1 - yp) + np.hypot(x2 - xp, y2 - yp) - np.hypot(x1 - x2, y1 - y2))
    
    
    def circle_point(self, xc: float, yc: float, r: float, xp: float, yp: float):
        return r >= np.hypot(xp - xc, yp - yc)
    
    
    def line_circle(self, x1: float, y1: float, x2: float, y2: float, xc: float, yc: float, r: float):
        if (self.circle_point(xc, yc, r, x1, y1) or self.circle_point(xc, yc, r, x2, y2)):
            return True
        dot = (((xc - x1) * (x2 - x1)) + ((yc - y1) * (y2 - y1))) / ((x1 - x2)**2 + (y1 - y2)**2)
        xnear = x1 + dot * (x2 - x1)
        ynear = y1 + dot * (y2 - y1)
        return self.circle_point(xc, yc, r, xnear, ynear) and self.line_point(x1, y1, x2, y2, xnear, ynear)
    
    
    def disk(self, robot, scale = 1):
        return (scale * (robot[0] - 3.2 * np.sin(robot[2])), scale * (robot[1] - 3.2 * np.cos(robot[2])))
    
    
    def collision_check(self):
        terminated = False
        positive = 0
        negative = 0
        p3 = self.disk(self._agent_location)
        p6 = self.disk(self._adversary_location)
        for i in range(4):
            i1 = i / 2
            i2 = (i1 + 0.5) % 2
            p1 = self.corner(self._agent_location, i1)
            p2 = self.corner(self._agent_location, i2)
            p4 = self.corner(self._adversary_location, i1)
            p5 = self.corner(self._adversary_location, i2)
            if self.line_circle(p1[0], p1[1], p2[0], p2[1], p6[0], p6[1], 3.25):
                negative = 1
                terminated = True
            if self.line_circle(p4[0], p4[1], p5[0], p5[1], p3[0], p3[1], 3.25):
                positive = 1
                terminated = True
        reward = positive - negative
#        if (reward == 0 and terminated) or self.circle_point(p3[0], p3[1], 6.5, p6[0], p6[1]):
#            return 0.1, True
        return reward, terminated
        
    def translate(self, robot, action, path_steps = 1):
        direction = self._action_to_direction[action] * np.array([1 / path_steps, 1, 1])
        return np.clip(np.array([robot[0] + direction[1] / path_steps * np.cos(robot[2] + np.pi/2), robot[1] - direction[1] / path_steps * np.sin(robot[2] + np.pi/2), robot[2]]) if direction[0] == 0 else np.array([robot[0] + direction[1] * (np.cos(direction[0] + robot[2] + direction[2]) - np.cos(robot[2] + direction[2])), robot[1] + direction[1] * -(np.sin(direction[0] + robot[2] + direction[2]) - np.sin(robot[2] + direction[2])), (robot[2] + direction[0])%(2 * np.pi)]), [6.45, 6.45, 0], [89.55, 89.55, 2 * np.pi])
        
    def aggressive_select(self):
        dist = 2000
        adv_act = -1
        for index in range(len(self._action_to_direction)):
            test = self.translate(self._adversary_location, index)
            temp_dist = np.hypot(self._agent_location[0] - self.disk(test)[0], self._agent_location[1] - self.disk(test)[1])
            if (temp_dist < dist):
                dist = temp_dist
                adv_act = index
        return adv_act
    
    def defensive_select(self):
        test = self.translate(self._adversary_location, 8)
        if (self._adversary_location != test).any() and self.line_circle(self._adversary_location[0], self._adversary_location[1], test[0], test[1], self._agent_location[0], self._agent_location[1], 3.9105):
            test = self.translate(self._adversary_location, 8, 20)
        temp_angle = np.absolute(2*np.pi - (test[2]+np.pi/2)%(2*np.pi) - np.arctan2(self._agent_location[1] - test[1], self._agent_location[0] - test[0])%(2*np.pi))
        angle = min(temp_angle, 2*np.pi - temp_angle)
        adv_act = 8
        for index in range(len(self._action_to_direction) - 1):
            test = self.translate(self._adversary_location, index)
            temp_angle = np.absolute(2*np.pi - (test[2]+np.pi/2)%(2*np.pi) - np.arctan2(self._agent_location[1] - test[1], self._agent_location[0] - test[0])%(2*np.pi))
            if (min(temp_angle, 2*np.pi - temp_angle) < angle):
                angle = min(temp_angle, 2*np.pi - temp_angle)
                adv_act = index
        return adv_act
            
    def render(self):
        if self.render_mode == "rgb_array" or self.render_mode == "human":
            return self._render_frame(self.text)

    def _render_frame(self, text):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        agent_corners = []
        adversary_corners = []
        for i in range(4):
            agent_corners.append(self.corner(self._agent_location, i / 2, 10.417))
            adversary_corners.append(self.corner(self._adversary_location, i / 2, 10.417))
        
        canvas = pygame.Surface((self.window_size, self.window_size))
        text = pygame.font.Font(size = 100).render(text, False, "black")
        canvas.fill("grey")
        
        pygame.draw.lines(canvas, "blue", True, agent_corners, 3)
        pygame.draw.lines(canvas, "red", True, adversary_corners, 3)
        pygame.draw.circle(canvas, "blue", self.disk(self._agent_location, 10.417), 33.333)
        pygame.draw.circle(canvas, "red", self.disk(self._adversary_location, 10.417), 33.333)
        
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            self.window.blit(text, text.get_rect(center = (500,500)))
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.fps)
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
