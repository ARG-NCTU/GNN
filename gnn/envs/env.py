import pygame
import gym
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import math
import csv

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
            "veh_collision": [],
            "pos": [], 
            "goal": [], 
            "vel": [], 
            "goal_index": [], 
            "local_fov": []
        }

        # environment params
        self.training_enable = True
        self.env_grid_px_per_m = self.cfg.env_px / self.cfg.env_dim[0]
    
    def initialize(self):
        self.observatiopn_space["veh_type"] = np.zeros((self.cfg.usv_agent_num + self.cfg.uav_agent_num, 2))
        self.observatiopn_space["veh_collision"] = np.zeros(self.cfg.usv_agent_num + self.cfg.uav_agent_num)
        self.observatiopn_space["vel"] = np.zeros(self.cfg.usv_agent_num + self.cfg.uav_agent_num)
        self.observatiopn_space["goal_index"] = np.zeros(self.cfg.goal_num)
        self.observatiopn_space["local_fov"] = np.zeros((self.cfg.usv_agent_num + self.cfg.uav_agent_num, self.cfg.usv_sense_radius*2+2, self.cfg.usv_sense_radius*2+2, 3))

        for agent in range(self.cfg.usv_agent_num + self.cfg.uav_agent_num):
            if agent < self.cfg.usv_agent_num:
                self.observatiopn_space["veh_type"][agent] = ENCODE_USV
                self.observatiopn_space["veh_collision"][agent] = self.cfg.usv_collision_radius
                self.observatiopn_space["vel"][agent] = self.cfg.usv_max_v
            else:
                self.observatiopn_space["veh_type"][agent] = ENCODE_UAV
                self.observatiopn_space["veh_collision"][agent] = self.cfg.uav_collision_radius
                self.observatiopn_space["vel"][agent] = self.cfg.uav_max_v
        
        self.loadMap()
        
        collision_check = True
        while collision_check == True:
            self.randomSpawnAgent()
            collision_check = False
            for agent in range(self.cfg.usv_agent_num + self.cfg.uav_agent_num):
                if self.agentCollisionUpdate(agent) == True:
                    collision_check = True
                    break
        return
    
    def fetchState(self):
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
    
    def getAgentFOV(self, agent_index):
        return self.observatiopn_space["local_fov"][agent_index]
        # return self.display_fov
    

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
            

    def manualUpdate(self, agent_index):
        self.observatiopn_space["pos"][agent_index][0] += self.keyboard_input[0]*self.observatiopn_space["vel"][agent_index]
        self.observatiopn_space["pos"][agent_index][1] += self.keyboard_input[1]*self.observatiopn_space["vel"][agent_index]
        self.keyboard_input[0] = 0
        self.keyboard_input[1] = 0
        # print(self.observatiopn_space["pos"][agent_index])
        return

    def goalUpdate(self): 
        idx = 0
        for goal in self.observatiopn_space["goal"]:
            for agent in self.observatiopn_space["pos"]:
                if self.distanceP2P(goal, agent) <= self.cfg.goal_margin:
                    self.observatiopn_space["goal_index"][idx] = 1
                    # print("REACH")
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
        self.observatiopn_space["goal_index"] = np.zeros(self.cfg.goal_num)
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
    
    def goalReached(self, goal_index):
        if self.observatiopn_space["goal_index"][goal_index] == False:
            return DISPLAY_UNREACH_GOAL
        else:
            return DISPLAY_REACHED_GOAL
    
    def agentLocalPerception(self, agent_index):
        pos = [
            round(self.observatiopn_space["pos"][agent_index][0]), 
            round(self.observatiopn_space["pos"][agent_index][1])
        ]
        
        fov_radius = self.cfg.usv_sense_radius

        for i in range(fov_radius*2):
            for j in range(fov_radius*2):
                map_x = round(self.cfg.env_dim[0]/2)-int(pos[1])+i-fov_radius
                map_y = round(self.cfg.env_dim[1]/2)-int(pos[0])+j-fov_radius
                if (map_x >= 0 and map_x < self.cfg.env_dim[0]) and (map_y >= 0 and map_y < self.cfg.env_dim[1]):
                    self.display_fov[i+1][j+1] = self.display_map[map_x][map_y]
                    self.data_fov[i+1][j+1] = self.data_map[map_x][map_y]
                else:
                    self.display_fov[i+1][j+1] = DISPLAY_UNKNOWN
                    self.data_fov[i+1][j+1] = DATA_UNKNOWN
         
                        
        # return perception_map_2d
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
            if self.observatiopn_space["goal_index"][idx] == False:
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

    def agentPerceptionUpdate(self, agent_index):
        # for i in range(self.cfg.usv_agent_num):
            #     self.agentGlobalPerception(i)
            #     self.agentLocalPerception(i)
        self.agentGlobalPerception(agent_index)
        self.agentLocalPerception(agent_index)

        fov_radius = self.cfg.usv_sense_radius
        old_fov = self.data_fov.copy()
        for i in range(fov_radius*2+2):
            for j in range(fov_radius*2+2):
                self.data_fov[i][j] = old_fov[j][i]

                # pass
        # for i in range(fov_radius*2+2):
        #     print(self.data_fov[i])
        # print("\n\n\n\n")
        self.observatiopn_space["local_fov"][agent_index] = self.display_fov
        return
    
    def agentCollisionUpdate(self, agent_index):
        veh_pos = [
            self.observatiopn_space["pos"][agent_index][0], 
            self.observatiopn_space["pos"][agent_index][1]
        ]
        veh_type = self.observatiopn_space["veh_type"][agent_index]
        veh_collision_radius = self.observatiopn_space["veh_collision"][agent_index]
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

                # debug
                # tmp = []
                # for i in range(self.cfg.usv_sense_radius*2+2):
                #     for j in range(self.cfg.usv_sense_radius*2+2):
                #         tmp[i][j] = (1, 2, 3)
                # self.observatiopn_space["local_fov"][agent] = tmp
                # debug

                print(np.size(self.display_fov))
                print(np.size(self.observatiopn_space["local_fov"][agent]))
                self.observatiopn_space["local_fov"][agent] = self.display_fov
                if self.agentCollisionUpdate(agent):
                    print("fuck")

            #   debug usage
            self.manualUpdate(monitor_agent)
            self.agentPerceptionUpdate(monitor_agent)
            test = self.agentCollisionUpdate(monitor_agent)
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
                    self.goalReached(i), 
                    self.transToDispaly(
                        self.observatiopn_space["goal"][i][0],
                        self.observatiopn_space["goal"][i][1]
                    ), 
                    self.cfg.goal_margin * self.env_grid_px_per_m
                )
            
            # display agent
            for agent in range(self.cfg.usv_agent_num + self.cfg.uav_agent_num):
                # print("number: ", i+1)
                # print(self.observatiopn_space["pos"][i])
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
            pygame.surfarray.blit_array(local_view, self.display_fov)
            local_view = pygame.transform.scale(local_view, (self.cfg.debug_view_px, self.cfg.debug_view_px))
            screen.blit(local_view, (self.cfg.env_px, 0))
            
            

            pygame.display.update()

        pygame.quit()







