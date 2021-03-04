#!/usr/bin/env python

import numpy as np
from .Agent import Agent 
import torch

COOP = 0
DEFECT = 1

class MGAdaptiveEnv(object):

    def __init__(self, args, policy_candidates = None):
        self.num_agents = 1
        self.agents = [Agent(args.history_length) for i in range(self.num_agents)]
        self.opponent = Agent(args.history_length)
        self.current_time = 0
        self.coop_num = 0
        self.defect_num = 0
        self.coopdefect_num = 0
        self.defectcoop_num = 0
        self.history_length = args.history_length
        self.game_length = args.game_length
        self.critic_full_obs = args.critic_full_obs
        self.num_policy_candidates = args.num_policy_candidates
        self.policy_candidates = policy_candidates
        assert self.history_length <= self.game_length
        
        if args.env_name == "StagHuntAdaptive":
            # CC: 4, CD: 0, DC: 3, DD: 1
            self.reward_matrix = [[0.4, -5.0], [0.3, 0.1]]
        elif args.env_name == "PrisonerDilemmaAdaptive":
            # CC: 3, CD: 0, DC: 4, DD: 1
            self.reward_matrix = [[0.3, 0.0], [0.4, 0.1]]
        elif args.env_name == "ChickenAdaptive":
            # CC: 3, CD: 1, DC: 4, DD: 0
            self.reward_matrix = [[0.3, 0.1], [0.4, 0.0]]
        else:
            raise NotImplementedError
        
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            self.action_space.append(agent.action_space)
            if self.critic_full_obs:
                agent.observation_space = [self.history_length * 2 + self.num_policy_candidates]
            self.observation_space.append(agent.observation_space)
        
        self.share_reward = args.share_reward
        self.shape_reward = args.shape_reward
        self.shape_beta = args.shape_beta
        self.limit_game = args.limit_game

    def reset(self, choose = -1):
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
        # opponent reset
        if choose == -1:
            self.select_opponent = np.random.randint(0, self.num_policy_candidates)
        else:
            self.select_opponent = choose
            
        self.opponent_actor_critic = self.policy_candidates[self.select_opponent]
               
        self.opponent.state = [-1] * self.history_length
        self.opponent.reward = 0
        self.opponent.done = False   
            
        self.agents[0].obs = np.array(self.agents[0].state + self.opponent.state)
        self.opponent.obs = np.array(self.opponent.state + self.agents[0].state)
        
        select_opponent_obs = np.zeros(self.num_policy_candidates)
        select_opponent_obs[self.select_opponent]=1

        self.recurrent_hidden_states = torch.zeros(1, self.opponent_actor_critic.recurrent_hidden_state_size)
        self.recurrent_hidden_states_critic = torch.zeros(1, self.opponent_actor_critic.recurrent_hidden_state_size)
        self.recurrent_c_states = torch.zeros(1, self.opponent_actor_critic.recurrent_hidden_state_size)
        self.recurrent_c_states_critic = torch.zeros(1, self.opponent_actor_critic.recurrent_hidden_state_size)
        self.masks = torch.ones(1,1)       
        # env obs
        obs_n = []
        obs_critic_n = []
        opponent_obs_n = []
        opponent_obs_critic_n = []
        if self.critic_full_obs:
            for agent in self.agents:
                obs_n.append(np.append(agent.obs, np.zeros(self.num_policy_candidates)))
                obs_critic = np.append(agent.obs, select_opponent_obs)
                obs_critic_n.append(obs_critic)
                opponent_obs_n.append(agent.obs)
                
            #obs_n.append(np.append(self.opponent.obs,np.zeros(self.num_policy_candidates)))
            opponent_obs_n.append(self.opponent.obs)
            #obs_critic_n.append(np.append(self.opponent.obs, select_opponent_obs))
            self.opponent_share_obs = torch.tensor(np.array(opponent_obs_n).reshape(1, -1),dtype=torch.float32)
            self.opponent_obs = torch.tensor(np.array([opponent_obs_n])[:,1,:],dtype=torch.float32)
            return obs_n, obs_critic_n, self.select_opponent                
        else:
            for agent in self.agents:
                obs_n.append(agent.obs) 
                opponent_obs_n.append(agent.obs)
            #obs_n.append(self.opponent.obs) 
            opponent_obs_n.append(self.opponent.obs) 
            self.opponent_share_obs = torch.tensor(np.array(opponent_obs_n).reshape(1, -1),dtype=torch.float32)
            self.opponent_obs = torch.tensor(np.array([opponent_obs_n])[:,1,:],dtype=torch.float32)        
            return obs_n, self.select_opponent

    def step(self, action_n):
        obs_n = []
        obs_critic_n = []
        opponent_obs_n = []
        opponent_obs_critic_n = []
        reward_n = []
        done_n = []
        info_n = {'coop&coop_num': [], 'defect&defect_num': [], 'coop&defect_num': [], 'defect&coop_num': []}

        self.current_time += 1
        # actions
        for i, agent in enumerate(self.agents):
            agent.action = np.argmax(action_n[i])
            agent.state.pop(0)
            agent.state.append(agent.action)
            agent.done = self.current_time >= self.game_length if self.limit_game else False
        
        _, a, _, recurrent_hidden_states, recurrent_hidden_states_critic, recurrent_c_states, recurrent_c_states_critic = self.opponent_actor_critic.act(self.opponent_share_obs, self.opponent_obs,self.recurrent_hidden_states,self.recurrent_hidden_states_critic,self.recurrent_c_states,self.recurrent_c_states_critic,self.masks,deterministic=True)

        self.recurrent_hidden_states = recurrent_hidden_states
        self.recurrent_hidden_states_critic = recurrent_hidden_states_critic
        self.recurrent_c_states = recurrent_c_states
        self.recurrent_c_states_critic = recurrent_c_states_critic

        self.opponent.action = int(a.cpu().numpy())
        # obs
        self.opponent.state.pop(0)
        self.opponent.state.append(self.opponent.action)
        self.opponent.done = self.current_time >= self.game_length if self.limit_game else False
        self.agents[0].obs = np.array(self.agents[0].state + self.opponent.state)  
        self.opponent.obs = np.array(self.opponent.state + self.agents[0].state)  
        # reward       
        self.agents[0].reward = self.reward_matrix[self.agents[0].action][self.opponent.action]
        self.opponent.reward = self.reward_matrix[self.opponent.action][self.agents[0].action]

        if self.agents[0].action == COOP and self.opponent.action == COOP:
            self.coop_num += 1
        elif self.agents[0].action == DEFECT and self.opponent.action == DEFECT:
            self.defect_num += 1
        elif self.agents[0].action == COOP and self.opponent.action == DEFECT:
            self.coopdefect_num += 1
        else:
            self.defectcoop_num += 1
            
        select_opponent_obs = np.zeros(self.num_policy_candidates)
        select_opponent_obs[self.select_opponent]=1
        
        for agent in self.agents:             
            if self.critic_full_obs:
                obs_n.append(np.append(agent.obs, np.zeros(self.num_policy_candidates)))
                obs_critic_n.append(np.append(agent.obs, select_opponent_obs))
                #obs_n.append(np.append(self.opponent.obs,np.zeros(self.num_policy_candidates)))
                #obs_critic_n.append(np.append(self.opponent.obs, select_opponent_obs))
                opponent_obs_n.append(agent.obs)
                opponent_obs_n.append(self.opponent.obs)
                self.opponent_share_obs = torch.tensor(np.array(opponent_obs_n).reshape(1, -1),dtype=torch.float32)
                self.opponent_obs = torch.tensor(np.array([opponent_obs_n])[:,1,:],dtype=torch.float32)
            else:
                obs_n.append(agent.obs)
                #obs_n.append(self.opponent.obs)
                opponent_obs_n.append(agent.obs)
                opponent_obs_n.append(self.opponent.obs)
                self.opponent_share_obs = torch.tensor(np.array(opponent_obs_n).reshape(1, -1),dtype=torch.float32)
                self.opponent_obs = torch.tensor(np.array([opponent_obs_n])[:,1,:],dtype=torch.float32)
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

        if self.critic_full_obs:
            return obs_n, obs_critic_n, self.select_opponent, reward_n, done_n, info_n
        else:
            return obs_n, self.select_opponent, reward_n, done_n, info_n
        
    def seed(self, seed):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)
    def close(self):
        self.agents = []
        return None
    
