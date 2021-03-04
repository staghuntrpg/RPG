#!/usr/bin/env python

import numpy as np
from .Agent import Agent

COOP = 0
DEFECT = 1

class MGEnv(object):

    def __init__(self, args,choose=0):
        self.num_agents = args.num_agents
        self.agents = [Agent(args.history_length) for i in range(self.num_agents)]
        self.current_time = 0
        self.coop_num = 0
        self.defect_num = 0
        self.coopdefect_num = 0
        self.defectcoop_num = 0
        self.history_length = args.history_length
        self.game_length = args.game_length
        self.reward_randomization = args.reward_randomization
        assert self.history_length <= self.game_length

        if args.env_name == "StagHunt":
            # CC: 4, CD: -100, DC: 3, DD: 1
            self.reward_matrix = [[0.4, -5.0], [0.3, 0.1]]
        elif args.env_name == "PrisonerDilemma":
            # CC: 3, CD: 0, DC: 4, DD: 1
            self.reward_matrix = [[0.3, 0.0], [0.4, 0.1]]
        elif args.env_name == "Chicken":
            # CC: 3, CD: 1, DC: 4, DD: 0
            self.reward_matrix = [[0.3, 0.1], [0.4, 0.0]]
        else:
            raise NotImplementedError

        if self.reward_randomization:
            reward = [[[0.4, 0.0], [0.0, 0.0]],
                      [[0.0, 0.0], [0.0, 0.4]],
                      [[0.0, 0.4], [0.4, 0.0]],
                      [[0.4, 0.1], [0.4, 0.0]]]            
            self.reward_matrix = reward[choose]    
            
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            self.action_space.append(agent.action_space)
            self.observation_space.append(agent.observation_space)
        
        self.share_reward = args.share_reward
        self.shape_reward = args.shape_reward
        self.shape_beta = args.shape_beta
        self.limit_game = args.limit_game

    def reset(self):
        self.current_time = 0
        self.coop_num = 0
        self.defect_num = 0
        self.coopdefect_num = 0
        self.defectcoop_num = 0

        # agent reset
        for agent in self.agents:
            agent.state = [-1] * self.history_length
            agent.reward = 0
            agent.done = False
        self.agents[0].obs = np.array(self.agents[0].state + self.agents[1].state)
        self.agents[1].obs = np.array(self.agents[1].state + self.agents[0].state)
        
        # env obs
        obs_n = []
        for agent in self.agents:
            obs_n.append(agent.obs)

        return obs_n

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'coop&coop_num': [], 'defect&defect_num': [], 'coop&defect_num': [], 'defect&coop_num': []}

        self.current_time += 1
        for i, agent in enumerate(self.agents):
            agent.action = np.argmax(action_n[i])
            agent.state.pop(0)
            agent.state.append(agent.action)
            agent.done = self.current_time >= self.game_length if self.limit_game else False

        self.agents[0].obs = np.array(self.agents[0].state + self.agents[1].state)  
        self.agents[1].obs = np.array(self.agents[1].state + self.agents[0].state)         
        self.agents[0].reward = self.reward_matrix[self.agents[0].action][self.agents[1].action]
        self.agents[1].reward = self.reward_matrix[self.agents[1].action][self.agents[0].action]

        if self.agents[0].action == COOP and self.agents[1].action == COOP:
            self.coop_num += 1
        elif self.agents[0].action == DEFECT and self.agents[1].action == DEFECT:
            self.defect_num += 1
        elif self.agents[0].action == COOP and self.agents[1].action == DEFECT:
            self.coopdefect_num += 1
        else:
            self.defectcoop_num += 1
           
        for agent in self.agents:   
            obs_n.append(agent.obs)
            reward_n.append(agent.reward)
            done_n.append(agent.done)
            
        info_n['coop&coop_num'] = self.coop_num
        info_n['defect&defect_num'] = self.defect_num
        info_n['coop&defect_num'] = self.coopdefect_num
        info_n['defect&coop_num'] = self.defectcoop_num
        # import pdb; pdb.set_trace()

        global_reward = np.sum(reward_n)
        
        if self.share_reward:
            reward_n = [global_reward] * self.num_agents
        
        if self.shape_reward:
            reward_n = list(map(lambda x :x[0] * self.shape_beta + x[1] * (1-self.shape_beta), zip([global_reward] * self.num_agents, reward_n)))
            
        return obs_n, reward_n, done_n, info_n
        
    def seed(self, seed):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)
            
    def close(self):
        self.agents = []
        return None
    
