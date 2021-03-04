import gym
from gym import error, spaces, utils
import numpy as np
import os
from .Agent import Agent
import torch
import matplotlib.pyplot as plt

ACTIONS = {'MOVE_LEFT': [0, -1],  # Move left
           'MOVE_RIGHT': [0, 1],  # Move right
           'MOVE_UP': [-1, 0],  # Move up
           'MOVE_DOWN': [1, 0],  # Move down
           'STAY': [0, 0]  # don't move
           }

# bgr
DEFAULT_COLOURS = {' ': [0, 0, 0],  # Black background
                   'S': [101, 67, 254],  # stag
                   'H': [178, 196, 47],  # hare1
                   'G': [178, 196, 47],  # hare2
                   'Y': [216, 30, 54],  # young
                   'M': [159, 67, 255],  # mature
                   'C': [238, 133, 114],  # chonghe
                   'D': [238, 133, 114],  # chonghe2
                   'E': [101, 67, 254],  # ecalation

                   # Colours for agents. R value is a unique identifier
                   '1': [166, 90, 3],  
                   '2': [30, 191, 252],  # Blue
                   '3': [204, 168, 0],
                   '4': [154, 157, 252]} 
TRUN_MATURE = 0.3
TRUN_DEATH = 0.3

class GridWorldAdaptiveEnv(object):
    def __init__(self, args, policy_candidates, length = 5):
        self.env_name = args.env_name
        self.num_agents = args.num_agents
        self.episode_length = args.episode_length
        self.length = length      
        self.color_map = DEFAULT_COLOURS          
        self.share_reward = args.share_reward        
        self.shape_reward = args.shape_reward
        self.shape_beta = args.shape_beta
        if self.env_name == "StagHuntGWAdaptive":
            self.coop = 5
            self.defect = 2
            self.gore = -2                
        elif self.env_name == "HarvestGWAdaptive":
            self.coop = 2
            self.defect = 1
        elif self.env_name == "EscalationGWAdaptive":
            self.coop = 1
            self.defect_coef = -0.9
            self.coop_length = 0     
        self.max_life = 20
        self.coop_num = 0
        self.gore1_num = 0
        self.gore2_num = 0
        self.hare1_num = 0
        self.hare2_num = 0
        self.num_policy_candidates = args.num_policy_candidates
        self.policy_candidates = policy_candidates
        self.critic_full_obs = args.critic_full_obs
        self.reset_map()
        self.setup_agents()       

    @property
    def action_space(self):
        action_space = []
        for agent in self.agents:
            action_space.append(agent.action_space)
        return action_space

    @property
    def observation_space(self):
        observation_space = []
        for agent in self.agents:
            if self.critic_full_obs:
                if 'StagHuntGW' in self.env_name:
                    agent.observation_space = [10 + self.num_policy_candidates]
                elif 'EscalationGW' in self.env_name:
                    agent.observation_space = [7 + self.num_policy_candidates]
            observation_space.append(agent.observation_space)
        return observation_space

    def StagHunt_setup_map(self):
        self.agents_start_pos = []
        self.stag_points = 0
        self.hare1_points = 0
        self.hare2_points = 0
        points = []
        num_index = 0
        while num_index < (self.num_agents+3):
            index = np.random.randint(0, self.length, (2)).tolist()
            if (index in points):
                continue
            else:
                points.append(index)
                num_index += 1
        
        for i in range(self.num_agents):
            self.agents_start_pos.append(points[i])

        self.base_map[points[-3][0], points[-3][1]] = 'S'
        self.stag_points += 1
        self.stag_pos = np.array(points[-3])
        self.base_map[points[-2][0], points[-2][1]] = 'H'
        self.hare1_points += 1
        self.hare1_pos = np.array(points[-2])
        self.base_map[points[-1][0], points[-1][1]] = 'G'
        self.hare2_points += 1
        self.hare2_pos = np.array(points[-1])

    def Harvest_setup_map(self):
        self.agents_start_pos = []
        self.life = np.zeros((self.length, self.length))
        self.young_points = 0
        self.young_pos = []
        self.mature_points = 0
        self.mature_pos = []
        points = []
        num_index = 0
        while num_index < (self.num_agents+1):
            index = np.random.randint(0, self.length, (2)).tolist()
            if index in points:
                continue
            else:
                points.append(index)
                num_index += 1
        
        for i in range(self.num_agents):
            self.agents_start_pos.append(points[i])

        self.base_map[points[-1][0], points[-1][1]] = 'Y'
        self.young_points += 1
        self.young_pos.append(points[-1])
        self.life[points[-1][0], points[-1][1]] += 1
        
    def Escalation_setup_map(self):
        self.agents_start_pos = []        
        points = []
        num_index = 0
        while num_index < (self.num_agents+1):
            index = np.random.randint(0, self.length, (2)).tolist()
            if (index in points):
                continue
            else:
                points.append(index)
                num_index += 1
        
        for i in range(self.num_agents):
            self.agents_start_pos.append(points[i])

        self.base_map[points[-1][0], points[-1][1]] = 'E'
        self.escalation_points = 1
        self.escalation_pos = np.array(points[-1])

    def setup_agents(self,choose = -1):
        self.coop_num = 0
        self.gore1_num = 0
        self.gore2_num = 0
        self.hare1_num = 0
        self.hare2_num = 0
        self.coop_length = 0
        self.agents = []
        self.opponent = None
        for i in range(self.num_agents):
            agent = Agent(i, self.agents_start_pos[i], self.base_map, self.env_name)
            self.agents.append(agent)
        # opponent reset
        print(choose)
        if choose == -1:
            self.select_opponent = np.random.randint(0, self.num_policy_candidates)
        else:
            self.select_opponent = choose
        print(self.select_opponent)
        self.opponent_actor_critic = self.policy_candidates[self.select_opponent]
        self.recurrent_hidden_states = torch.zeros(1, self.opponent_actor_critic.recurrent_hidden_state_size)
        self.recurrent_hidden_states_critic = torch.zeros(1, self.opponent_actor_critic.recurrent_hidden_state_size)
        self.recurrent_c_states = torch.zeros(1, self.opponent_actor_critic.recurrent_hidden_state_size)
        self.recurrent_c_states_critic = torch.zeros(1, self.opponent_actor_critic.recurrent_hidden_state_size)
        self.masks = torch.ones(1,1)

    def map_to_colors(self, base_map=None, color_map=None):
        """Converts a map to an array of RGB values.
        Parameters
        ----------
        map: np.ndarray
            map to convert to colors
        color_map: dict
            mapping between array elements and desired colors
        Returns
        -------
        arr: np.ndarray
            3-dim numpy array consisting of color map
        """
        if base_map is None:
            base_map = self.get_map_with_agents()
        if color_map is None:
            color_map = self.color_map

        rgb_arr = np.zeros((base_map.shape[0], base_map.shape[1], 3), dtype=int)
        for row_elem in range(base_map.shape[0]):
            for col_elem in range(base_map.shape[1]):
                rgb_arr[row_elem, col_elem, :] = color_map[base_map[row_elem, col_elem]]

        return rgb_arr

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        if 'Harvest' in self.env_name:
            rgb_arr = self.map_to_colors(self.get_map_with_agents(), self.color_map)
            return rgb_arr.transpose(2,0,1)
        elif 'StagHunt' in self.env_name:
            # my pos
            my_pos = self.agents[agent_id].pos.tolist()
            # other pos
            other_pos = self.agents[1-agent_id].pos.tolist()
            # stag_pos
            stag_pos = self.stag_pos.tolist()
            # plant_pos
            hare1_pos = self.hare1_pos.tolist()
            hare2_pos = self.hare2_pos.tolist()
            return np.concatenate([my_pos]+[other_pos]+[stag_pos]+[hare1_pos]+[hare2_pos])
        elif 'Escalation' in self.env_name:
            # my pos
            my_pos = self.agents[agent_id].pos.tolist()
            # other pos
            other_pos = self.agents[1-agent_id].pos.tolist()
            # escalation pos
            escalation_pos = self.escalation_pos.tolist()            
            #return np.concatenate([my_pos]+[other_pos]+[escalation_pos]+[[self.coop_length]])
            return np.concatenate([my_pos]+[other_pos]+[escalation_pos])

    def reset_map(self):
        """Resets the map to be empty as well as a custom reset set by subclasses"""
        self.base_map = np.full((self.length, self.length),' ')
        if self.env_name == "StagHuntGWAdaptive":
            self.StagHunt_setup_map()
        elif self.env_name == "HarvestGWAdaptive":
            self.Harvest_setup_map()
        elif self.env_name == "EscalationGWAdaptive":
            self.Escalation_setup_map()
        

    def get_map_with_agents(self):
        """Gets a version of the environment map where generic
        'P' characters have been replaced with specific agent IDs.

        Returns:
            2D array of strings representing the map.
        """
        map_with_agents = np.copy(self.base_map)

        for i in range(self.num_agents):
            char_id = str(i + 1) # agent-i
            if map_with_agents[self.agents[i].pos[0], self.agents[i].pos[1]] == ' ':
                map_with_agents[self.agents[i].pos[0], self.agents[i].pos[1]] = char_id
            else:
                map_with_agents[self.agents[i].pos[0], self.agents[i].pos[1]] = '3'

        return map_with_agents
   
    def update_moves(self, agent_actions):
        for agent_id, action in agent_actions.items():
            agent = self.agents[agent_id]            
            selected_action = ACTIONS[action]
            new_pos = agent.get_pos() + selected_action
            # allow the agents to confirm what position they can move to               
            agent.update_agent_pos(new_pos)

    def update_stag(self):
        if self.stag_points > 0:
            minimum = 100
            for i in range(self.num_agents):
                x = self.agents[i].get_pos() - self.stag_pos
                dist = np.sum(np.square(x))
                if dist < minimum:
                    minimum = dist
                    minimum_index = i
            # move
            x = self.agents[minimum_index].get_pos() - self.stag_pos
            sign_x = np.sign(x)
            if 0 in sign_x:
                if sign_x[0]==0 and sign_x[1]==0:
                    pass
                else:
                    temp_pos = self.stag_pos + sign_x 
                    if self.base_map[temp_pos[0],temp_pos[1]] == ' ':
                        if self.base_map[self.stag_pos[0],self.stag_pos[1]] == 'C':
                            self.base_map[self.stag_pos[0],self.stag_pos[1]] = 'H'
                        elif self.base_map[self.stag_pos[0],self.stag_pos[1]] == 'D':
                            self.base_map[self.stag_pos[0],self.stag_pos[1]] = 'G'
                        else:
                            self.base_map[self.stag_pos[0],self.stag_pos[1]] = ' '
                        
                        self.stag_pos = temp_pos
                        self.base_map[self.stag_pos[0],self.stag_pos[1]] = 'S'
                    else:
                        if self.base_map[self.stag_pos[0],self.stag_pos[1]] == 'C':
                            self.base_map[self.stag_pos[0],self.stag_pos[1]] = 'H'
                        elif self.base_map[self.stag_pos[0],self.stag_pos[1]] == 'D':
                            self.base_map[self.stag_pos[0],self.stag_pos[1]] = 'G'
                        else:
                            self.base_map[self.stag_pos[0],self.stag_pos[1]] = ' '
                        self.stag_pos = temp_pos
                        if self.base_map[self.stag_pos[0],self.stag_pos[1]] == 'H':
                            self.base_map[self.stag_pos[0],self.stag_pos[1]] = 'C'
                        elif self.base_map[self.stag_pos[0],self.stag_pos[1]] == 'G':
                            self.base_map[self.stag_pos[0],self.stag_pos[1]] = 'D'
            else:
                temp_sign = np.copy(sign_x)
                choose = np.random.randint(0,2)
                temp_sign[choose] = 0
                temp_pos = self.stag_pos + temp_sign
                if self.base_map[temp_pos[0],temp_pos[1]] == 'H' or self.base_map[temp_pos[0],temp_pos[1]] == 'G':
                    temp_sign1 = np.copy(sign_x)
                    temp_sign1[1-choose] = 0
                    temp_pos1 = self.stag_pos + temp_sign1 
                    if self.base_map[temp_pos1[0],temp_pos1[1]] == ' ':                    
                        if self.base_map[self.stag_pos[0],self.stag_pos[1]] == 'C':
                            self.base_map[self.stag_pos[0],self.stag_pos[1]] = 'H'
                        elif self.base_map[self.stag_pos[0],self.stag_pos[1]] == 'D':
                            self.base_map[self.stag_pos[0],self.stag_pos[1]] = 'G'
                        else:
                            self.base_map[self.stag_pos[0],self.stag_pos[1]] = ' '
                        self.stag_pos = temp_pos1
                        self.base_map[self.stag_pos[0],self.stag_pos[1]] = 'S'
                    else:
                        if self.base_map[self.stag_pos[0],self.stag_pos[1]] == 'C':
                            self.base_map[self.stag_pos[0],self.stag_pos[1]] = 'H'
                        elif self.base_map[self.stag_pos[0],self.stag_pos[1]] == 'D':
                            self.base_map[self.stag_pos[0],self.stag_pos[1]] = 'G'
                        else:
                            self.base_map[self.stag_pos[0],self.stag_pos[1]] = ' '
                        self.stag_pos = temp_pos1
                        if self.base_map[self.stag_pos[0],self.stag_pos[1]] == 'H':
                            self.base_map[self.stag_pos[0],self.stag_pos[1]] = 'C'
                        elif self.base_map[self.stag_pos[0],self.stag_pos[1]] == 'G':
                            self.base_map[self.stag_pos[0],self.stag_pos[1]] = 'D'
                else:
                    if self.base_map[self.stag_pos[0],self.stag_pos[1]] == 'C':
                        self.base_map[self.stag_pos[0],self.stag_pos[1]] = 'H'
                    elif self.base_map[self.stag_pos[0],self.stag_pos[1]] == 'D':
                        self.base_map[self.stag_pos[0],self.stag_pos[1]] = 'G'
                    else:
                        self.base_map[self.stag_pos[0],self.stag_pos[1]] = ' '
                    self.stag_pos = temp_pos
                    self.base_map[self.stag_pos[0],self.stag_pos[1]] = 'S'

    def StagHuntUpdateMap(self):
        while self.stag_points < 1:
            index = np.random.randint(0, self.length, (2)).tolist()
            map_with_agents = self.get_map_with_agents()
            if map_with_agents[index[0], index[1]] == ' ':
                self.base_map[index[0], index[1]] = 'S'
                self.stag_points += 1
                self.stag_pos = np.array(index)
        while self.hare1_points < 1:
            index = np.random.randint(0, self.length, (2)).tolist()
            if self.get_map_with_agents()[index[0], index[1]] == ' ':
                self.base_map[index[0], index[1]] = 'H'
                self.hare1_points += 1
                self.hare1_pos = np.array(index)
        while self.hare2_points < 1:
            index = np.random.randint(0, self.length, (2)).tolist()
            if self.get_map_with_agents()[index[0], index[1]] == ' ':
                self.base_map[index[0], index[1]] = 'G'
                self.hare2_points += 1
                self.hare2_pos = np.array(index)

    def HarvestUpdateMap(self):
        for i in range(self.life.shape[0]):
            for j in range(self.life.shape[1]):
                if self.base_map[i,j] == 'Y' or self.base_map[i,j] == 'M':
                    self.life[i][j] += 1

        # mature to death
        old_mature_pos = np.copy(self.mature_pos)
        for i, mature in enumerate(old_mature_pos):
            if self.life[mature[0],mature[1]] >= self.max_life or np.random.rand(1)[0] <= TRUN_DEATH:
                self.mature_points -= 1
                self.mature_pos.remove(mature.tolist())
                self.base_map[mature[0],mature[1]] = ' '
                self.life[mature[0],mature[1]] = 0
                
        # young to mature
        old_young_pos = np.copy(self.young_pos)
        for i, young in enumerate(old_young_pos):
            if self.life[young[0],young[1]] >= (self.max_life-1) or np.random.rand(1)[0] <= TRUN_MATURE:
                self.base_map[young[0], young[1]] = 'M'
                self.mature_points += 1
                self.mature_pos.append(young.tolist())
                self.young_points -= 1
                self.young_pos.remove(young.tolist())
                
        num_plants_old = self.young_points + self.mature_points
        map_with_agents = self.get_map_with_agents()

        # young born
        if num_plants_old < 4:
            while (self.young_points + self.mature_points) - num_plants_old < 1:
                index = np.random.randint(0, self.length, (2)).tolist()
                if map_with_agents[index[0], index[1]] == ' ':
                    self.base_map[index[0], index[1]] = 'Y'
                    self.young_points += 1
                    self.young_pos.append(index)
                    self.life[index[0],index[1]] += 1
                    
                    
    def EscalationUpdateMap(self):
        actions = [[0,1],[0,-1],[-1,0],[1,0]]
        last_pos = self.escalation_pos      
        while self.escalation_points == 0:
            next_choose = np.random.randint(0,4)          
            next_pos = last_pos + actions[next_choose]
            next_row, next_col = next_pos
            if next_row < 0 or next_row >= self.length or next_col < 0 or next_col >= self.length:
                self.escalation_points = 0
            else:
                self.base_map[last_pos[0], last_pos[1]] = ' '
                self.escalation_points = 1
                self.escalation_pos = next_pos
                self.base_map[next_pos[0], next_pos[1]] = 'E'

    def HarvestConsume(self, pos0, pos1):
        """Defines how an agent interacts with the char it is standing on"""
        charA = self.base_map[pos0[0], pos0[1]]
        charB = self.base_map[pos1[0], pos1[1]]
        if pos0==pos1:
            if charA == 'M':
                self.agents[0].reward_this_turn += self.coop
                self.agents[1].reward_this_turn += self.coop
                self.mature_points -= 1
                self.mature_pos.remove(pos0)
                self.base_map[pos0[0], pos0[1]] = ' '
                self.life[pos0[0], pos0[1]] = 0
            elif charA == 'Y':
                index = np.random.randint(0,2)
                self.agents[index].reward_this_turn += self.defect
                self.young_points -= 1
                self.young_pos.remove(pos0)
                self.base_map[pos0[0], pos0[1]] = ' '
                self.life[pos0[0], pos0[1]] = 0
        else:        
            if charA == 'M':
                self.agents[0].reward_this_turn += self.coop
                self.agents[1].reward_this_turn += self.coop
                self.mature_points -= 1
                self.mature_pos.remove(pos0)
                self.base_map[pos0[0], pos0[1]] = ' '
                self.life[pos0[0], pos0[1]] = 0
                if charB == 'M':
                    self.agents[0].reward_this_turn += self.coop
                    self.agents[1].reward_this_turn += self.coop
                    self.mature_points -= 1
                    self.mature_pos.remove(pos1)
                    self.base_map[pos1[0], pos1[1]] = ' '
                    self.life[pos1[0], pos1[1]] = 0
                elif charB == 'Y':
                    self.agents[1].reward_this_turn += self.defect
                    self.young_points -= 1
                    self.young_pos.remove(pos1)
                    self.base_map[pos1[0], pos1[1]] = ' '
                    self.life[pos1[0], pos1[1]] = 0
            elif charA == 'Y':
                self.agents[0].reward_this_turn += self.defect
                self.young_points -= 1
                self.young_pos.remove(pos0)
                self.base_map[pos0[0], pos0[1]] = ' '
                self.life[pos0[0], pos0[1]] = 0
                if charB == 'M':
                    self.agents[0].reward_this_turn += self.coop
                    self.agents[1].reward_this_turn += self.coop
                    self.mature_points -= 1
                    self.mature_pos.remove(pos1)
                    self.base_map[pos1[0], pos1[1]] = ' '
                    self.life[pos1[0], pos1[1]] = 0
                elif charB == 'Y':
                    self.agents[1].reward_this_turn += self.defect
                    self.young_points -= 1
                    self.young_pos.remove(pos1)
                    self.base_map[pos1[0], pos1[1]] = ' '
                    self.life[pos1[0], pos1[1]] = 0
            elif charA == ' ':
                if charB == 'M':
                    self.agents[0].reward_this_turn += self.coop
                    self.agents[1].reward_this_turn += self.coop
                    self.mature_points -= 1
                    self.mature_pos.remove(pos1)
                    self.base_map[pos1[0], pos1[1]] = ' '
                    self.life[pos1[0], pos1[1]] = 0
                elif charB == 'Y':
                    self.agents[1].reward_this_turn += self.defect
                    self.young_points -= 1
                    self.young_pos.remove(pos1)
                    self.base_map[pos1[0], pos1[1]] = ' '
                    self.life[pos1[0], pos1[1]] = 0
    
    def StagHuntConsume(self, pos0, pos1):
        """Defines how an agent interacts with the char it is standing on"""
        charA = self.base_map[pos0[0], pos0[1]]
        charB = self.base_map[pos1[0], pos1[1]]
        if charA == 'S':
            if charB == 'S':
                self.coop_num += 1
                self.agents[0].reward_this_turn += self.coop
                self.agents[1].reward_this_turn += self.coop
                self.stag_points -= 1
                self.base_map[pos0[0], pos0[1]] = ' '
            elif charB == 'H':
                self.gore1_num += 1
                self.agents[0].reward_this_turn += self.gore
                self.stag_points -= 1
                self.base_map[pos0[0], pos0[1]] = ' '
                self.agents[1].reward_this_turn += self.defect
                self.hare1_points -= 1
                self.hare2_num += 1
                self.base_map[pos1[0], pos1[1]] = ' '
            elif charB == 'G':
                self.gore1_num += 1
                self.agents[0].reward_this_turn += self.gore
                self.stag_points -= 1
                self.base_map[pos0[0], pos0[1]] = ' '
                self.agents[1].reward_this_turn += self.defect
                self.hare2_points -= 1
                self.hare2_num += 1
                self.base_map[pos1[0], pos1[1]] = ' '
            else:
                self.gore1_num += 1
                self.agents[0].reward_this_turn += self.gore
                self.stag_points -= 1
                self.base_map[pos0[0], pos0[1]] = ' '
        elif charA == 'H':
            if charB == 'S':
                self.gore2_num += 1
                self.agents[0].reward_this_turn += self.defect
                self.hare1_num += 1
                self.hare1_points -= 1
                self.base_map[pos0[0], pos0[1]] = ' '
                self.agents[1].reward_this_turn += self.gore
                self.stag_points -= 1
                self.base_map[pos1[0], pos1[1]] = ' '
            elif charB == 'D':
                self.gore2_num += 1
                self.agents[0].reward_this_turn += self.defect
                self.hare1_num += 1
                self.hare1_points -= 1
                self.base_map[pos0[0], pos0[1]] = ' '
                self.agents[1].reward_this_turn += self.gore
                self.agents[1].reward_this_turn += self.defect
                self.hare2_num += 1
                self.stag_points -= 1
                self.hare2_points -= 1
                self.base_map[pos1[0], pos1[1]] = ' '
            elif charB == 'G':
                self.agents[0].reward_this_turn += self.defect
                self.hare1_num += 1
                self.hare1_points -= 1
                self.base_map[pos0[0], pos0[1]] = ' '
                self.agents[1].reward_this_turn += self.defect
                self.hare2_num += 1
                self.hare2_points -= 1
                self.base_map[pos1[0], pos1[1]] = ' '
            elif charB == 'H':
                self.hare1_points -= 1
                self.base_map[pos0[0], pos0[1]] = ' '
                index = np.random.randint(0,2)
                self.agents[index].reward_this_turn += self.defect
                if index == 0:
                    self.hare1_num += 1
                else:
                    self.hare2_num += 1                    
            else:
                self.agents[0].reward_this_turn += self.defect
                self.hare1_num += 1
                self.hare1_points -= 1
                self.base_map[pos0[0], pos0[1]] = ' '
        elif charA == 'G':
            if charB == 'S':
                self.gore2_num += 1
                self.agents[0].reward_this_turn += self.defect
                self.hare1_num += 1
                self.hare2_points -= 1
                self.base_map[pos0[0], pos0[1]] = ' '
                self.agents[1].reward_this_turn += self.gore
                self.stag_points -= 1
                self.base_map[pos1[0], pos1[1]] = ' '
            elif charB == 'C':
                self.gore2_num += 1
                self.agents[0].reward_this_turn += self.defect
                self.hare1_num += 1
                self.hare2_points -= 1
                self.base_map[pos0[0], pos0[1]] = ' '
                self.agents[1].reward_this_turn += self.gore
                self.agents[1].reward_this_turn += self.defect
                self.hare2_num += 1
                self.stag_points -= 1
                self.hare1_points -= 1
                self.base_map[pos1[0], pos1[1]] = ' '
            elif charB == 'H':
                self.agents[0].reward_this_turn += self.defect
                self.hare1_num += 1
                self.hare2_points -= 1
                self.base_map[pos0[0], pos0[1]] = ' '
                self.agents[1].reward_this_turn += self.defect
                self.hare2_num += 1
                self.hare1_points -= 1
                self.base_map[pos1[0], pos1[1]] = ' '
            elif charB == 'G':
                self.hare2_points -= 1
                self.base_map[pos0[0], pos0[1]] = ' '
                index = np.random.randint(0,2)
                self.agents[index].reward_this_turn += self.defect
                if index == 0:
                    self.hare1_num += 1
                else:
                    self.hare2_num += 1
            else:
                self.agents[0].reward_this_turn += self.defect
                self.hare1_num += 1
                self.hare2_points -= 1
                self.base_map[pos0[0], pos0[1]] = ' '
        elif charA == 'C':
            if charB == 'C':
                self.coop_num += 1
                self.agents[0].reward_this_turn += self.coop               
                self.agents[1].reward_this_turn += self.coop
                index = np.random.randint(0,2)
                self.agents[index].reward_this_turn += self.defect
                self.hare2_num += 1                
                self.stag_points -= 1
                self.hare1_points -= 1
                self.base_map[pos0[0], pos0[1]] = ' '                
            elif charB == 'G':
                self.gore1_num += 1
                self.agents[0].reward_this_turn += self.gore
                self.agents[0].reward_this_turn += self.defect
                self.hare1_num += 1
                self.stag_points -= 1
                self.hare1_points -= 1
                self.base_map[pos0[0], pos0[1]] = ' '                
                self.agents[1].reward_this_turn += self.defect
                self.hare2_num += 1
                self.hare2_points -= 1               
                self.base_map[pos1[0], pos1[1]] = ' '
            else:
                self.gore1_num += 1
                self.agents[0].reward_this_turn += self.gore
                self.agents[0].reward_this_turn += self.defect
                self.hare1_num += 1
                self.stag_points -= 1
                self.hare1_points -= 1
                self.base_map[pos0[0], pos0[1]] = ' '
        elif charA == 'D':
            if charB == 'D':
                self.coop_num += 1
                self.agents[0].reward_this_turn += self.coop               
                self.agents[1].reward_this_turn += self.coop
                index = np.random.randint(0,2)
                self.agents[index].reward_this_turn += self.defect
                if index == 0:
                    self.hare1_num += 1
                else:
                    self.hare2_num += 1                
                self.stag_points -= 1
                self.hare2_points -= 1
                self.base_map[pos0[0], pos0[1]] = ' '                
            elif charB == 'H':
                self.gore1_num += 1
                self.agents[0].reward_this_turn += self.gore
                self.agents[0].reward_this_turn += self.defect
                self.hare1_num += 1
                self.stag_points -= 1
                self.hare2_points -= 1
                self.base_map[pos0[0], pos0[1]] = ' '                
                self.agents[1].reward_this_turn += self.defect
                self.hare2_num += 1
                self.hare1_points -= 1               
                self.base_map[pos1[0], pos1[1]] = ' '
            else:
                self.gore1_num += 1
                self.agents[0].reward_this_turn += self.gore
                self.agents[0].reward_this_turn += self.defect
                self.hare1_num += 1
                self.stag_points -= 1
                self.hare2_points -= 1
                self.base_map[pos0[0], pos0[1]] = ' '
        elif charA == ' ':
            if charB == 'S':
                self.gore2_num += 1
                self.agents[1].reward_this_turn += self.gore
                self.stag_points -= 1
                self.base_map[pos1[0], pos1[1]] = ' '
            elif charB == 'H':
                self.agents[1].reward_this_turn += self.defect
                self.hare2_num += 1
                self.hare1_points -= 1
                self.base_map[pos1[0], pos1[1]] = ' '
            elif charB == 'G':
                self.agents[1].reward_this_turn += self.defect
                self.hare2_num += 1
                self.hare2_points -= 1
                self.base_map[pos1[0], pos1[1]] = ' '
            elif charB == 'C':
                self.gore2_num += 1
                self.agents[1].reward_this_turn += self.gore
                self.agents[1].reward_this_turn += self.defect
                self.hare2_num += 1
                self.stag_points -= 1
                self.hare1_points -= 1
                self.base_map[pos1[0], pos1[1]] = ' '
            elif charB == 'D':
                self.gore2_num += 1
                self.agents[1].reward_this_turn += self.gore
                self.agents[1].reward_this_turn += self.defect
                self.hare2_num += 1
                self.stag_points -= 1
                self.hare2_points -= 1
                self.base_map[pos1[0], pos1[1]] = ' '
                
    def EscalationConsume(self, pos0, pos1):
        charA = self.base_map[pos0[0], pos0[1]]
        charB = self.base_map[pos1[0], pos1[1]]

        if charA == 'E':
            if charB == 'E':
                self.agents[0].reward_this_turn += self.coop
                self.agents[1].reward_this_turn += self.coop
                self.coop_length += 1
                self.coop_num += 1
                self.escalation_points = 0
            else:
                if self.coop_length > 0:
                    self.agents[0].reward_this_turn = self.defect_coef * self.coop_length
                    self.agents[0].done = True
                    self.agents[1].done = True
        else:
            if charB == 'E' and self.coop_length > 0:
                self.agents[1].reward_this_return = self.defect_coef * self.coop_length
                self.agents[0].done = True
                self.agents[1].done = True
                      
    def close(self):
        self.agents = []
        return None

    def seed(self, seed):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def render(self, filename=None):
        """ Creates an image of the map to plot or save.

        Args:
            path: If a string is passed, will save the image
                to disk at this location.
        """
        map_with_agents = self.get_map_with_agents()

        rgb_arr = self.map_to_colors(map_with_agents)
        plt.figure()
        plt.imshow(rgb_arr, interpolation='nearest')        
        text = "coop_num = " + str(self.coop_num) + "/" + str(self.episode_length)        
        plt.text(0, 0, text, fontdict={'size': 10, 'color':  'white'})
        if 'StagHunt' in self.env_name: 
            text = "agent1_gore_num = " + str(self.gore1_num) + "/" + str(self.episode_length)
            plt.text(0, 0.2, text, fontdict={'size': 10, 'color':  'white'})
            text = "agent1_hare_num = " + str(self.hare1_num) + "/" + str(self.episode_length)
            plt.text(0, 0.4, text, fontdict={'size': 10, 'color':  'white'})
            text = "agent2_gore_num = " + str(self.gore2_num) + "/" + str(self.episode_length)
            plt.text(0, 0.6, text, fontdict={'size': 10, 'color':  'white'})
            text = "agent2_hare_num = " + str(self.hare2_num) + "/" + str(self.episode_length)
            plt.text(0, 0.8, text, fontdict={'size': 10, 'color':  'white'})
        plt.title(self.env_name + "_Game: Policy-" + str(self.select_opponent) )       
        if filename is not None:
            plt.savefig(filename)
        
        return rgb_arr.astype(np.uint8)

    def step(self, actions): #action [1,2,4,3,7]
        """A single environment step. Returns reward, terminated, info."""
        actions = [np.argmax(a) for a in actions]

        _, a, _, recurrent_hidden_states, recurrent_hidden_states_critic, recurrent_c_states, recurrent_c_states_critic = self.opponent_actor_critic.act(self.opponent_share_obs, self.opponent_obs,self.recurrent_hidden_states,self.recurrent_hidden_states_critic,self.recurrent_c_states,self.recurrent_c_states_critic,self.masks)

        self.recurrent_hidden_states = recurrent_hidden_states
        self.recurrent_hidden_states_critic = recurrent_hidden_states_critic
        self.recurrent_c_states = recurrent_c_states
        self.recurrent_c_states_critic = recurrent_c_states_critic
               
        actions.append(int(a.cpu().numpy()))

        agent_actions = {}
        for i in range(self.num_agents):
            agent_action = self.agents[i].action_map(actions[i]) # such as 'FIRE'
            agent_actions[i] = agent_action

        # stag
        if self.env_name == "StagHuntGWAdaptive":
            self.update_stag()
        # move
        self.update_moves(agent_actions)
                   
        pos0 = self.agents[0].get_pos().tolist()
        pos1 = self.agents[1].get_pos().tolist()
        if self.env_name == 'StagHuntGWAdaptive':
            self.StagHuntConsume(pos0, pos1)
            self.StagHuntUpdateMap()
        elif self.env_name == 'HarvestGWAdaptive':
            self.HarvestConsume(pos0, pos1)
            self.HarvestUpdateMap()
        elif self.env_name == 'EscalationGWAdaptive':
            self.EscalationConsume(pos0, pos1)
            self.EscalationUpdateMap()

        observations = []
        critic_observations = []
        opponent_observations = []
        rewards = []
        dones = []
        infos = {'collective_return': [], 'coop&coop_num': [], 'gore1_num': [], 'gore2_num': [],'hare1_num': [], 'hare2_num': []}
        
        select_opponent_obs = np.zeros(self.num_policy_candidates).tolist()
        select_opponent_obs[self.select_opponent]=1
        
        for i in range(self.num_agents):
            obs = self.get_obs_agent(i)
            opponent_observations.append(obs)                        
            if self.critic_full_obs:
                critic_observations.append(np.concatenate([obs]+[select_opponent_obs]))
                observations.append(np.concatenate([obs]+[np.zeros(self.num_policy_candidates).tolist()]))
            else:
                observations.append(obs)
            reward = self.agents[i].compute_reward() * 0.1
            rewards.append(reward)
            dones.append(self.agents[i].get_done())
            
        self.opponent_share_obs = torch.tensor(np.array([opponent_observations]).reshape(1, -1),dtype=torch.float32)
        self.opponent_obs = torch.tensor(np.array([opponent_observations])[:,1,:],dtype=torch.float32)
            
        collective_return = 0
        for i in range(self.num_agents):
            collective_return += self.agents[i].collective_return
            
        if rewards[0] > 0.1 and rewards[1] > 0.1:
            self.coop_num += 1

        infos['collective_return'] = collective_return
        infos['coop&coop_num'] = self.coop_num
        if 'StagHunt' in self.env_name:
            infos['gore1_num'] = self.gore1_num
            infos['gore2_num'] = self.gore2_num
            infos['hare1_num'] = self.hare1_num
            infos['hare2_num'] = self.hare2_num
            
        global_reward = np.sum(rewards)  
        if self.share_reward:
            rewards = [global_reward] * self.num_agents

        if self.shape_reward:
            rewards = list(map(lambda x :x[0] * self.shape_beta + x[1] * (1-self.shape_beta), zip([global_reward] * self.num_agents, rewards)))
            
        if self.critic_full_obs:
            return observations, critic_observations, self.select_opponent, rewards, dones, infos
        else:
            return observations, self.select_opponent, rewards, dones, infos

    def reset(self, choose=-1):
        """Reset the environment. Required after each full episode.
        Returns initial observations and states.
        """   
        self.reset_map()
        print("choose:")
        print(choose)
        self.setup_agents(choose=choose)
        
        select_opponent_obs = np.zeros(self.num_policy_candidates).tolist()
        select_opponent_obs[self.select_opponent]=1

        observations = []
        critic_observations = []
        opponent_observations = []
        
        for i in range(self.num_agents):
            obs = self.get_obs_agent(i)
            if self.critic_full_obs:
                observations.append(np.concatenate([obs]+[np.zeros(self.num_policy_candidates).tolist()]))
                critic_observations.append(np.concatenate([obs]+[select_opponent_obs]))
            else:
                observations.append(obs)
            opponent_observations.append(obs)
            
                   
        self.opponent_share_obs = torch.tensor(np.array([opponent_observations]).reshape(1, -1),dtype=torch.float32)
        self.opponent_obs = torch.tensor(np.array([opponent_observations])[:,1,:],dtype=torch.float32)
        
        if self.critic_full_obs:
            return observations, critic_observations, self.select_opponent
        else:
            return observations, self.select_opponent



    