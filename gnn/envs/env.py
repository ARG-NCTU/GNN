import pygame
import gym
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import math
import csv

# log usage
from datetime import datetime
# log usage

DISPLAY_UNKNOWN = (0, 0, 0)
DISPLAY_WHITE = (255, 255, 255)
DISPLAY_WATER = (14, 180, 255)
DISPLAY_LAND = (160, 60, 10)
DISPLAY_USV_AGENT = (30, 5, 230)
DISPLAY_UAV_AGENT = (250, 205, 30)
DISPLAY_UNREACH_GOAL = (250, 15, 30)
DISPLAY_REACHED_GOAL = (0, 190, 50)

DATA_UNKNOWN = 0
DATA_WHITE = 0
DATA_WATER = 0
DATA_LAND = 1
DATA_USV_AGENT = 2
DATA_UNREACH_GOAL = 3
DATA_REACHED_GOAL = 4

ENCODE_USV = (1, 0)
ENCODE_UAV = (0, 1)



DISPLAY_MAP = {
    0: DISPLAY_WATER,
    1: DISPLAY_LAND,
}

DATA_MAP = {
    0: DATA_WATER,
    1: DATA_LAND,
}


class EnvManager(gym.Env):
    def __init__(self, args):
        self.cfg = args
        self.keyboard_input = [0, 0]
        self.display_fov = np.zeros((self.cfg.usv_sense_radius*2+2, self.cfg.usv_sense_radius*2+2, 3))
        self.data_fov = np.zeros((self.cfg.usv_sense_radius*2+2, self.cfg.usv_sense_radius*2+2))

        self.display_map = np.zeros((self.cfg.env_dim[1], self.cfg.env_dim[0], 3))
        self.data_map = np.zeros((self.cfg.env_dim[1], self.cfg.env_dim[0]))

        
        # gym requirement 
        self.usv_action_space = gym.spaces.Tuple(
            (
                gym.spaces.Box(
                    low = -self.cfg.usv_max_v,
                    high = self.cfg.usv_max_v,
                    shape = (2,),
                    dtype = float,
                ),
            ) * self.cfg.usv_agent_num
        )
        self.uav_action_space = gym.spaces.Tuple(
            (
                gym.spaces.Box(
                    low = -self.cfg.uav_max_v,
                    high = self.cfg.uav_max_v,
                    shape = (2,),
                    dtype = float,
                ),
            ) * self.cfg.uav_agent_num
        )

        self.observatiopn_space = {
            "veh_type": [], 
            "collision_radius": [],
            "sense_radius": [],
            "pos": [], 
            "goal": [], 
            "vel": [], 
            "goal_status": [], 
            "local_fov": []
        }

        # environment params
        self.training_enable = True
        self.env_grid_px_per_m = self.cfg.env_px / self.cfg.env_dim[0]

    def printLog(self, msg):
        currentDateAndTime = datetime.now()
        print("[{time}]{info}".format(time = currentDateAndTime, info = msg))
        return

    def initialize(self): # declare observation space size & load map
        self.observatiopn_space["veh_type"] = np.zeros((self.cfg.usv_agent_num + self.cfg.uav_agent_num, 2))
        self.observatiopn_space["collision_radius"] = np.zeros(self.cfg.usv_agent_num + self.cfg.uav_agent_num)
        self.observatiopn_space["vel"] = np.zeros(self.cfg.usv_agent_num + self.cfg.uav_agent_num)
        self.observatiopn_space["goal_status"] = np.zeros(self.cfg.goal_num)
        self.observatiopn_space["sense_radius"] = np.zeros(self.cfg.usv_agent_num + self.cfg.uav_agent_num)
        self.observatiopn_space["local_fov"] = np.zeros((self.cfg.usv_agent_num + self.cfg.uav_agent_num, self.cfg.usv_sense_radius*2+2, self.cfg.usv_sense_radius*2+2, 3))

        for agent in range(self.cfg.usv_agent_num + self.cfg.uav_agent_num):
            if agent < self.cfg.usv_agent_num:
                self.observatiopn_space["veh_type"][agent] = ENCODE_USV
                self.observatiopn_space["collision_radius"][agent] = self.cfg.usv_collision_radius
                self.observatiopn_space["sense_radius"][agent] = self.cfg.usv_sense_radius
                self.observatiopn_space["vel"][agent] = self.cfg.usv_max_v # manual 
            else:
                self.observatiopn_space["veh_type"][agent] = ENCODE_UAV
                self.observatiopn_space["collision_radius"][agent] = self.cfg.uav_collision_radius
                self.observatiopn_space["sense_radius"][agent] = self.cfg.uav_sense_radius
                self.observatiopn_space["vel"][agent] = self.cfg.uav_max_v # manual 
        
        self.loadMap()

        return

    def reset(self): # radom assign agent & goal position, return observation space
        collision_check = True
        while collision_check == True:
            self.randomSpawnAgent()
            collision_check = False
            for agent in range(self.cfg.usv_agent_num + self.cfg.uav_agent_num):
                if self.agentCollisionUpdate(agent) == True:
                    collision_check = True
                    break

        for agent in range(self.cfg.usv_agent_num + self.cfg.uav_agent_num):
            self.agentPerceptionUpdate(agent)
        

        return self.observatiopn_space
    

    def loadMap(self):
        raw_map_array = []
        with open('./envs/map.csv', newline='') as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                raw_map_array.append(row)
        
        for i in range(self.cfg.env_dim[0]):
                for j in range(self.cfg.env_dim[1]):
                    self.display_map[j][i] = DISPLAY_MAP[int(raw_map_array[i][j])]
                    self.data_map[j][i] = DATA_MAP[int(raw_map_array[i][j])]
        return

    def transToDispaly(self, standard_x, standard_y):
        pygame_x = round(self.cfg.env_px / 2 - standard_y * self.env_grid_px_per_m)
        pygame_y = round(self.cfg.env_px / 2 - standard_x * self.env_grid_px_per_m)
        transfer_tuple = (pygame_x, pygame_y)
        return transfer_tuple

    def transToStandard(self, pygame_x, pygame_y):
        standard_x = (self.cfg.env_px / 2 - pygame_y) / self.env_grid_px_per_m
        standard_y = (self.cfg.env_px / 2 - pygame_x) / self.env_grid_px_per_m
        transfer_tuple = (standard_x, standard_y)
        return transfer_tuple

    def distanceP2P(self, p1, p2):
        return math.sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))

    def getCollisionType(self, agent_index):
        if (self.observatiopn_space["veh_type"][agent_index] == ENCODE_USV).all():
            return DISPLAY_LAND
        else:
            return DISPLAY_LAND

    def getCollisionBody(self, agent_index, other_agent_pos, collision_radius):
        fov_radius = int(self.observatiopn_space["sense_radius"][agent_index])
        diameter = int(collision_radius*2)
        offset_radius = diameter/2-0.5
        pixels_in_line = 0
        pixels_per_line = []
        collision_gird_array = []
        for i in range(diameter):
            for j in range(diameter):
                x = i - offset_radius
                y = j - offset_radius
                if x * x + y * y <= offset_radius * offset_radius + 1:
                    pixels_in_line += 1
            pixels_per_line.append(pixels_in_line)
            pixels_in_line = 0

        for i in range(len(pixels_per_line)):
            grid_y = fov_radius-round(other_agent_pos[1])+(len(pixels_per_line)-1)/2-i
            for j in range(pixels_per_line[i]):
                grid_x = fov_radius-round(other_agent_pos[0])+(pixels_per_line[i]-1)/2-j
                collision_gird_array.append([grid_x, grid_y])
        return collision_gird_array
            
    def manualUpdate(self, agent_index):
        self.observatiopn_space["pos"][agent_index][0] += self.keyboard_input[0]*self.observatiopn_space["vel"][agent_index]
        self.observatiopn_space["pos"][agent_index][1] += self.keyboard_input[1]*self.observatiopn_space["vel"][agent_index]
        self.keyboard_input[0] = 0
        self.keyboard_input[1] = 0
        return

    def goalUpdate(self): 
        idx = 0
        for goal in self.observatiopn_space["goal"]:
            for agent in self.observatiopn_space["pos"]:
                if self.distanceP2P(goal, agent) <= self.cfg.goal_margin:
                    self.observatiopn_space["goal_status"][idx] = 1
            idx+=1
        return

    def randomSpawnAgent(self):
        random_agent = gym.spaces.Box(
            -min(self.cfg.env_dim[0]/2, self.cfg.env_dim[1]/2),
            min(self.cfg.env_dim[0]/2, self.cfg.env_dim[1]/2),
            shape=((self.cfg.usv_agent_num + self.cfg.uav_agent_num), 2), 
            dtype=float
        )

        random_goal = gym.spaces.Box(
            -min(self.cfg.env_dim[0]/2, self.cfg.env_dim[1]/2),
            min(self.cfg.env_dim[0]/2, self.cfg.env_dim[1]/2),
            shape=((self.cfg.goal_num), 2), 
            dtype=float
        )
        self.observatiopn_space["goal"] = random_goal.sample()
        self.observatiopn_space["pos"] = random_agent.sample()
        self.observatiopn_space["goal_status"] = np.zeros(self.cfg.goal_num)
        return

    def debugSample(self):
        print("\n")
        test1 = self.usv_action_space.sample()
        print(type(test1))
        print(test1)
        print(test1[0])
        print("-------------------")
        test = self.uav_action_space.sample()
        print(type(test))
        print(test)
        print(test[0])
        return
    
    def displayGoalStaus(self, goal_index):
        if self.observatiopn_space["goal_status"][goal_index] == False:
            return DISPLAY_UNREACH_GOAL
        else:
            return DISPLAY_REACHED_GOAL
    
    def agentLocalPerception(self, agent_index):
        pos = [
            self.observatiopn_space["pos"][agent_index][0], 
            self.observatiopn_space["pos"][agent_index][1]
        ]
        collision_radius = self.observatiopn_space["collision_radius"][agent_index]
        fov_radius = int(self.observatiopn_space["sense_radius"][agent_index])

        # check static obstacle
        for i in range(fov_radius*2):
            for j in range(fov_radius*2):
                map_x = round(self.cfg.env_dim[0]/2)-int(round(pos[1]))+i-fov_radius
                map_y = round(self.cfg.env_dim[1]/2)-int(round(pos[0]))+j-fov_radius
                if (map_x >= 0 and map_x < self.cfg.env_dim[0]) and (map_y >= 0 and map_y < self.cfg.env_dim[1]):
                    self.display_fov[i+1][j+1] = self.display_map[map_x][map_y]
                    self.data_fov[i+1][j+1] = self.data_map[map_x][map_y]
                else:
                    self.display_fov[i+1][j+1] = DISPLAY_UNKNOWN
                    self.data_fov[i+1][j+1] = DATA_UNKNOWN
        
        # check the other agents
        for agent in self.observatiopn_space["pos"]:   
            if (agent == pos).all():
                continue 
            else:
                if abs(round(pos[0])-round(agent[0])) <= fov_radius + int(round(collision_radius)) and \
                    abs(round(pos[1])-round(agent[1])) <= fov_radius + int(round(collision_radius)):

                    
                    err_x = round(agent[0]) - round(pos[0])
                    err_y = round(agent[1]) - round(pos[1])

                    collision_gird_array = self.getCollisionBody(agent_index, [err_x, err_y], collision_radius)
                    
                    for grid in collision_gird_array:
                        y = int(grid[1])
                        x = int(grid[0])
                        if y >= 1 and y < 2*fov_radius+1 and x >= 1 and x < 2*fov_radius+1:
                            self.display_fov[y][x] = DISPLAY_LAND
        return

    def agentGlobalPerception(self, agent_index):
        pos = [
            round(self.observatiopn_space["pos"][agent_index][0]), 
            round(self.observatiopn_space["pos"][agent_index][1])
        ]

        fov_radius = self.cfg.usv_sense_radius
        # place goal into FOV feature
        for i in range(fov_radius*2+2):
            for j in range(fov_radius*2+2):
                if i == 0 or i == fov_radius*2+1 or j == 0 or j == fov_radius*2+1:
                    self.display_fov[i][j] = DISPLAY_WATER
                    self.data_fov[i][j] = DATA_WATER
        
        for idx in range(self.cfg.goal_num):
            if self.observatiopn_space["goal_status"][idx] == False:
                err_x = self.observatiopn_space["goal"][idx][0] - pos[0]
                err_y = self.observatiopn_space["goal"][idx][1] - pos[1]
                if abs(err_x) >= abs(err_y): # front triangle & rear triangle
                    if err_x >= 0: # front
                        self.display_fov[fov_radius-round(fov_radius * err_y / err_x)][0] = DISPLAY_UNREACH_GOAL
                        self.data_fov[fov_radius-round(fov_radius * err_y / err_x)][0] = DATA_UNREACH_GOAL
                    else: # rear
                        self.display_fov[fov_radius-round(fov_radius * err_y / abs(err_x))][fov_radius*2+1] = DISPLAY_UNREACH_GOAL
                        self.data_fov[fov_radius-round(fov_radius * err_y / abs(err_x))][fov_radius*2+1] = DATA_UNREACH_GOAL
                else: # left triangle & right triangle
                    if err_y >= 0: # left
                        self.display_fov[0][fov_radius-round(fov_radius * err_x / err_y)] = DISPLAY_UNREACH_GOAL
                        self.data_fov[0][fov_radius-round(fov_radius * err_x / err_y)] = DATA_UNREACH_GOAL
                    else: # right
                        self.display_fov[fov_radius*2+1][fov_radius-round(fov_radius * err_x / abs(err_y))] = DISPLAY_UNREACH_GOAL
                        self.data_fov[fov_radius*2+1][fov_radius-round(fov_radius * err_x / abs(err_y))] = DATA_UNREACH_GOAL
                
        # place the other agent into FOV feature
        



        # return perception_map_2d
        return

    def agentPerceptionUpdate(self, agent_index): # 
        self.agentGlobalPerception(agent_index)
        self.agentLocalPerception(agent_index)

        fov_radius = self.cfg.usv_sense_radius
        old_fov = self.data_fov.copy()
        for i in range(fov_radius*2+2):
            for j in range(fov_radius*2+2):
                self.data_fov[i][j] = old_fov[j][i]

        self.observatiopn_space["local_fov"][agent_index] = self.display_fov
        return
    
    def agentCollisionUpdate(self, agent_index):
        veh_pos = [
            self.observatiopn_space["pos"][agent_index][0], 
            self.observatiopn_space["pos"][agent_index][1]
        ]
        veh_type = self.observatiopn_space["veh_type"][agent_index]
        veh_collision_radius = self.observatiopn_space["collision_radius"][agent_index]
        collision_object = self.getCollisionType(agent_index)
        check_bit = False

        # check environment collision
        for i in range(int(veh_collision_radius*2)):
            for j in range(int(veh_collision_radius*2)):
                
                map_x = int(round(self.cfg.env_dim[0]/2)-int(veh_pos[1])+i-veh_collision_radius)
                map_y = int(round(self.cfg.env_dim[1]/2)-int(veh_pos[0])+j-veh_collision_radius)

                if (map_x >= 0 and map_x < self.cfg.env_dim[0]) and (map_y >= 0 and map_y < self.cfg.env_dim[1]):
                    if (self.display_map[map_x][map_y] == collision_object).all():
                        check_bit = True  
                else:
                    check_bit = True
        
        # check agent collision
        for agent in range(self.cfg.usv_agent_num + self.cfg.uav_agent_num):
            compare_pos = [
                self.observatiopn_space["pos"][agent][0], 
                self.observatiopn_space["pos"][agent][1]
            ]
            if (agent_index != agent) and (self.observatiopn_space["veh_type"][agent] == veh_type).all():
                distance = self.distanceP2P(veh_pos, compare_pos)
                if distance <= veh_collision_radius*2:
                    check_bit = True

        if check_bit == True:
            # print("Collision")
            return True
        else:
            # print("Safe")
            return False

    def manualDisplay(self):
        pygame.init()
        screen = pygame.display.set_mode([self.cfg.env_px + self.cfg.debug_view_px, self.cfg.env_px])
        monitor_agent = 0
        while True:
            # event & input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        if monitor_agent < self.cfg.usv_agent_num + self.cfg.uav_agent_num - 1:
                            monitor_agent+=1
                        else:
                            monitor_agent = 0
                    if event.key == pygame.K_LEFT:
                        self.keyboard_input[1] += 1
                    if event.key == pygame.K_RIGHT:
                        self.keyboard_input[1] -= 1
                    if event.key == pygame.K_UP:
                        self.keyboard_input[0] += 1
                    if event.key == pygame.K_DOWN:
                        self.keyboard_input[0] -= 1

            # update information
            self.goalUpdate()
            
            for agent in range(self.cfg.usv_agent_num + self.cfg.uav_agent_num):
                self.agentPerceptionUpdate(agent)
                if self.agentCollisionUpdate(agent):
                    self.printLog("fuck, it crashed")

            #   debug usage
            self.manualUpdate(monitor_agent)
            #   debug usage

            # display global setting
            env = pygame.Surface((self.cfg.env_dim[0], self.cfg.env_dim[1]))
            pygame.surfarray.blit_array(env, self.display_map)
            env = pygame.transform.scale(env, (self.cfg.env_px, self.cfg.env_px))
            screen.blit(env, (0, 0))

            # display goal
            for i in range(self.cfg.goal_num):
                pygame.draw.circle(
                    screen, 
                    self.displayGoalStaus(i), 
                    self.transToDispaly(
                        self.observatiopn_space["goal"][i][0],
                        self.observatiopn_space["goal"][i][1]
                    ), 
                    self.cfg.goal_margin * self.env_grid_px_per_m
                )
            
            # display agent
            for agent in range(self.cfg.usv_agent_num + self.cfg.uav_agent_num):
                if (self.observatiopn_space["veh_type"][agent] == ENCODE_USV).all():
                    pygame.draw.circle(
                        screen, 
                        DISPLAY_USV_AGENT, 
                        self.transToDispaly(
                            self.observatiopn_space["pos"][agent][0],
                            self.observatiopn_space["pos"][agent][1]
                        ), 
                        self.cfg.usv_collision_radius * self.env_grid_px_per_m
                    )
                else:
                    pygame.draw.circle(
                        screen, 
                        DISPLAY_UAV_AGENT, 
                        self.transToDispaly(
                            self.observatiopn_space["pos"][agent][0],
                            self.observatiopn_space["pos"][agent][1]
                        ), 
                        self.cfg.uav_collision_radius * self.env_grid_px_per_m
                    )
            
            # display_fov
            local_view = pygame.Surface((self.cfg.usv_sense_radius*2+2, self.cfg.usv_sense_radius*2+2))
            pygame.surfarray.blit_array(local_view, self.observatiopn_space["local_fov"][monitor_agent])
            local_view = pygame.transform.scale(local_view, (self.cfg.debug_view_px, self.cfg.debug_view_px))
            screen.blit(local_view, (self.cfg.env_px, 0))
            
            
            pygame.display.update()

        pygame.quit()







