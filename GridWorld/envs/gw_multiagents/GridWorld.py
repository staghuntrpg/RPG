import numpy as np
from .Agent import Agent
import matplotlib.pyplot as plt
import random
ACTIONS = {'MOVE_LEFT': [0, -1],  # Move left
           'MOVE_RIGHT': [0, 1],  # Move right
           'MOVE_UP': [-1, 0],  # Move up
           'MOVE_DOWN': [1, 0],  # Move down
           'STAY': [0, 0]  # don't move
           }
# bgr
DEFAULT_COLOURS = {' ': [0, 0, 0],  # Black background

                   'Stag': [101, 67, 254],  # stag
                   'Hare1': [178, 196, 47],  # hare1
                   'Hare2': [178, 196, 47],  # hare2
                   'Stag_with_hare1': [238, 133, 114],  # stag and hare1 重合
                   'Stag_with_hare2': [238, 133, 114],  # stag and hare2 重合

                   # Colours for agents. R value is a unique identifier
                   '1': [166, 90, 3],  
                   '2': [30, 191, 252],  # Blue
                   '3': [204, 168, 0],
                   '4': [154, 157, 252]}
TRUN_MATURE = 0.3
TRUN_DEATH = 0.3 

class multi_GridWorldEnv(object):
    def __init__(self, args, choose=0, length = 5):
        self.env_name = args.env_name
        self.num_agents = args.num_agents
        self.episode_length = args.episode_length
        self.length = length      
        self.color_map = DEFAULT_COLOURS          
        self.share_reward = args.share_reward        
        self.shape_reward = args.shape_reward
        self.shape_beta = args.shape_beta
        if self.env_name == "multi_StagHuntGW":
            self.coop = 5  
            self.defect = 2
            self.gore = -2
            self.reward_randomization = args.reward_randomization
            if self.reward_randomization:
                coop = [5]
                defect = [0]
                gore = [0]

                self.coop = coop[choose]
                self.defect = defect[choose]
                self.gore = gore[choose]

        self.max_life = 20
        self.coop_num = 0       
        self.reset_map()
        self.setup_agents()

    def reset_map(self):
        """Resets the map to be empty as well as a custom reset set by subclasses"""
        self.base_map = np.full((self.length, self.length),' ')
        if self.env_name == "multi_StagHuntGW":
            self.StagHunt_setup_map()


    def setup_agents(self):
        self.coop_num = 0
        self.agents = []
        for i in range(self.num_agents):
            agent = Agent(i, self.agents_start_pos[i], self.base_map, self.env_name,self.num_agents)
            self.agents.append(agent)


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

        self.base_map[points[-3][0], points[-3][1]] = 'Stag'
        self.stag_points += 1
        self.stag_pos = np.array(points[-3])
        self.base_map[points[-2][0], points[-2][1]] = 'Hare1'
        self.hare1_points += 1
        self.hare1_pos = np.array(points[-2])
        #G=H2
        self.base_map[points[-1][0], points[-1][1]] = 'Hare2'
        self.hare2_points += 1
        self.hare2_pos = np.array(points[-1])



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
        if self.env_name == 'multi_StagHuntGW':
            # my pos
            my_pos = self.agents[agent_id].pos.tolist()
            # other pos
            other_pos=[]
            for i in range(self.num_agents):
                if i ==agent_id:
                    continue
                else:
                    other_pos.append(self.agents[i].pos.tolist())
            other_pos=np.concatenate(other_pos)

            # stag_pos
            stag_pos = self.stag_pos.tolist()
            # plant_pos
            hare1_pos = self.hare1_pos.tolist()
            hare2_pos = self.hare2_pos.tolist()
            return np.concatenate([my_pos]+[other_pos]+[stag_pos]+[hare1_pos]+[hare2_pos])
        

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
                        if self.base_map[self.stag_pos[0],self.stag_pos[1]] == 'Stag_with_hare1':
                            self.base_map[self.stag_pos[0],self.stag_pos[1]] = 'Hare1'
                        elif self.base_map[self.stag_pos[0],self.stag_pos[1]] == 'Stag_with_hare2':
                            self.base_map[self.stag_pos[0],self.stag_pos[1]] = 'Hare2'
                        else:
                            self.base_map[self.stag_pos[0],self.stag_pos[1]] = ' '
                        
                        self.stag_pos = temp_pos
                        self.base_map[self.stag_pos[0],self.stag_pos[1]] = 'Stag'
                    else:
                        if self.base_map[self.stag_pos[0],self.stag_pos[1]] == 'Stag_with_hare1':
                            self.base_map[self.stag_pos[0],self.stag_pos[1]] = 'Hare1'
                        elif self.base_map[self.stag_pos[0],self.stag_pos[1]] == 'Stag_with_hare1':
                            self.base_map[self.stag_pos[0],self.stag_pos[1]] = 'Hare2'
                        else:
                            self.base_map[self.stag_pos[0],self.stag_pos[1]] = ' '
                        self.stag_pos = temp_pos
                        if self.base_map[self.stag_pos[0],self.stag_pos[1]] == 'Hare1':
                            self.base_map[self.stag_pos[0],self.stag_pos[1]] = 'Stag_with_hare1'
                        elif self.base_map[self.stag_pos[0],self.stag_pos[1]] == 'Hare2':
                            self.base_map[self.stag_pos[0],self.stag_pos[1]] = 'Stag_with_hare2'
            else:
                temp_sign = np.copy(sign_x)
                choose = np.random.randint(0,2)
                temp_sign[choose] = 0
                temp_pos = self.stag_pos + temp_sign
                if self.base_map[temp_pos[0],temp_pos[1]] == 'Hare1' or self.base_map[temp_pos[0],temp_pos[1]] == 'Hare2':
                    temp_sign1 = np.copy(sign_x)
                    temp_sign1[1-choose] = 0
                    temp_pos1 = self.stag_pos + temp_sign1 
                    if self.base_map[temp_pos1[0],temp_pos1[1]] == ' ':                    
                        if self.base_map[self.stag_pos[0],self.stag_pos[1]] == 'Stag_with_hare1':
                            self.base_map[self.stag_pos[0],self.stag_pos[1]] = 'Hare1'
                        elif self.base_map[self.stag_pos[0],self.stag_pos[1]] == 'Stag_with_hare2':
                            self.base_map[self.stag_pos[0],self.stag_pos[1]] = 'Hare2'
                        else:
                            self.base_map[self.stag_pos[0],self.stag_pos[1]] = ' '
                        self.stag_pos = temp_pos1
                        self.base_map[self.stag_pos[0],self.stag_pos[1]] = 'Stag'
                    else:
                        if self.base_map[self.stag_pos[0],self.stag_pos[1]] == 'Stag_with_hare1':
                            self.base_map[self.stag_pos[0],self.stag_pos[1]] = 'Hare1'
                        elif self.base_map[self.stag_pos[0],self.stag_pos[1]] == 'Stag_with_hare2':
                            self.base_map[self.stag_pos[0],self.stag_pos[1]] = 'Hare2'
                        else:
                            self.base_map[self.stag_pos[0],self.stag_pos[1]] = ' '
                        self.stag_pos = temp_pos1
                        if self.base_map[self.stag_pos[0],self.stag_pos[1]] == 'Hare1':
                            self.base_map[self.stag_pos[0],self.stag_pos[1]] = 'Stag_with_hare1'
                        elif self.base_map[self.stag_pos[0],self.stag_pos[1]] == 'Hare2':
                            self.base_map[self.stag_pos[0],self.stag_pos[1]] = 'Stag_with_hare2'
                else:
                    if self.base_map[self.stag_pos[0],self.stag_pos[1]] == 'Stag_with_hare1':
                        self.base_map[self.stag_pos[0],self.stag_pos[1]] = 'Hare1'
                    elif self.base_map[self.stag_pos[0],self.stag_pos[1]] == 'Stag_with_hare2':
                        self.base_map[self.stag_pos[0],self.stag_pos[1]] = 'Hare2'
                    else:
                        self.base_map[self.stag_pos[0],self.stag_pos[1]] = ' '
                    self.stag_pos = temp_pos
                    self.base_map[self.stag_pos[0],self.stag_pos[1]] = 'Stag'

    def StagHuntUpdateMap(self):
        while self.stag_points < 1:
            index = np.random.randint(0, self.length, (2)).tolist()
            map_with_agents = self.get_map_with_agents()
            if map_with_agents[index[0], index[1]] == ' ':
                self.base_map[index[0], index[1]] = 'Stag'
                self.stag_points += 1
                self.stag_pos = np.array(index)
        while self.hare1_points < 1:
            index = np.random.randint(0, self.length, (2)).tolist()
            if self.get_map_with_agents()[index[0], index[1]] == ' ':
                self.base_map[index[0], index[1]] = 'Hare1'
                self.hare1_points += 1
                self.hare1_pos = np.array(index)
        while self.hare2_points < 1:
            index = np.random.randint(0, self.length, (2)).tolist()
            if self.get_map_with_agents()[index[0], index[1]] == ' ':
                self.base_map[index[0], index[1]] = 'Hare2'
                self.hare2_points += 1
                self.hare2_pos = np.array(index)

    
    def StagHuntConsume(self, pos):
        """Defines how an agent interacts with the char it is standing on"""
        char_agents=[]
        for i in range(self.num_agents ):
            char_agents.append(self.base_map[pos[i][0],pos[i][1]])

        char_stag=self.base_map[self.stag_pos[0],self.stag_pos[1]]

        indexs = [i for i in range(self.num_agents)]
        #coop
        coop_flag=True
        for i in range(self.num_agents-1):
            if pos[i]!=pos[i+1] or pos[i]!=list(self.stag_pos):
                coop_flag=False

        if coop_flag:
            self.coop_num += 1
            for i in range(self.num_agents):
                self.agents[i].reward_this_turn += self.coop
            self.stag_points -= 1
            self.base_map[pos[0][0], pos[0][1]] = ' '
            if char_stag =='Stag_with_hare1':
                self.hare1_points -= 1
                index = np.random.randint(0, self.num_agents)
                self.agents[index].reward_this_turn += self.defect
                self.agents[index].hare_num += 1
            if char_stag =='Stag_with_hare2':
                self.hare2_points -= 1
                index = np.random.randint(0, self.num_agents)
                self.agents[index].reward_this_turn += self.defect
                self.agents[index].hare_num += 1
        else:
            hare1 = True
            hare2 = True
            random.shuffle(indexs)  # 确保随机吃到hare

            for index in indexs:
                if pos[index]==list(self.stag_pos):
                    self.agents[index].reward_this_turn += self.gore
                    self.agents[index].gore_num+=1
                    self.stag_points -= 1
                    self.base_map[self.stag_pos[0], self.stag_pos[1]] = ' '
                if pos[index]==list(self.hare1_pos) and hare1==True:
                    hare1=False
                    self.agents[index].reward_this_turn += self.defect
                    self.agents[index].hare_num += 1
                    self.hare1_points -= 1
                    self.base_map[self.hare1_pos[0], self.hare1_pos[1]] = ' '
                if pos[index] == list(self.hare2_pos) and hare2 == True:
                    hare2 = False
                    self.agents[index].reward_this_turn += self.defect
                    self.agents[index].hare_num += 1
                    self.hare2_points -= 1
                    self.base_map[self.hare2_pos[0], self.hare2_pos[1]] = ' '


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
        
        if 'StagHunt' in self.env_name: 
            text = "#Coop.-Hunt = " + str(self.coop_num) + "/" + str(self.episode_length)        
            plt.text(0, 0, text, fontdict={'size': 10, 'color':  'white'})

            k=[self.agents[i].gore_num for i in range(self.num_agents)]
            k=sum(k)
            text = "#Single-Hunt = " + str(k) + "/" + str(self.episode_length)
            plt.text(0, 0.2, text, fontdict={'size': 10, 'color':  'white'})

            k = [self.agents[i].hare_num for i in range(self.num_agents)]
            k = sum(k)
            text = "#Apple = " + str(k) + "/" + str(self.episode_length)
            plt.text(0, 0.4, text, fontdict={'size': 10, 'color':  'white'})
            #text = "agent2_gore_num = " + str(self.gore2_num) + "/" + str(self.episode_length)
            #plt.text(0, 0.6, text, fontdict={'size': 10, 'color':  'white'})
            #text = "agent2_hare_num = " + str(self.hare2_num) + "/" + str(self.episode_length)
            #plt.text(0, 0.8, text, fontdict={'size': 10, 'color':  'white'})       
            plt.title("Monster-Hunt") 
        elif 'Escalation' in self.env_name: 
            text = "#Coop. Length L = " + str(self.coop_num) + "/" + str(self.episode_length)        
            plt.text(0, 0, text, fontdict={'size': 10, 'color':  'white'})
            plt.title("Escalation") 
            
        if filename is not None:
            plt.savefig(filename)
        
        return rgb_arr.astype(np.uint8)

    def step(self, actions): #action [1,2,4,3,7]
        """A single environment step. Returns reward, terminated, info."""
        actions = [np.argmax(a) for a in actions]
        agent_actions = {}
        for i in range(self.num_agents):
            agent_action = self.agents[i].action_map(actions[i]) 
            agent_actions[i] = agent_action
        if self.env_name == 'multi_StagHuntGW':
            self.update_stag()
               
        # move
        self.update_moves(agent_actions)

        pos=[]
        for i in range(self.num_agents):
            pos.append(self.agents[i].get_pos().tolist())

        if self.env_name == 'multi_StagHuntGW':
            self.StagHuntConsume(pos)
            self.StagHuntUpdateMap()


        observations = []
        rewards = []
        dones = []
        infos = {'collective_return': [], 'coop&coop_num': []}
        for i in range(self.num_agents):
            key='gore{0}_num'.format(i)
            infos.update({key:[]})
            key = 'hare{0}_num'.format(i)
            infos.update({key: []})

        for i in range(self.num_agents):
            observations.append(self.get_obs_agent(i))
            reward = self.agents[i].compute_reward() * 0.1
            rewards.append(reward)
            dones.append(self.agents[i].get_done())
            
        collective_return = 0
        for i in range(self.num_agents):
            collective_return += self.agents[i].collective_return

        infos['collective_return'] = collective_return
        infos['coop&coop_num'] = self.coop_num
        if self.env_name == 'multi_StagHuntGW':
            for i in range(self.num_agents):
                infos['gore{0}_num'.format(i)] = self.agents[i].gore_num
                infos['hare{0}_num'.format(i)] = self.agents[i].hare_num

            
        global_reward = np.sum(rewards)  
        if self.share_reward:
            rewards = [global_reward] * self.num_agents

        if self.shape_reward:
            rewards = list(map(lambda x :x[0] * self.shape_beta + x[1] * (1-self.shape_beta), zip([global_reward] * self.num_agents, rewards)))

        return observations, rewards, dones, infos

    def reset(self):
        """Reset the environment. Required after each full episode.
        Returns initial observations and states.
        """   
        self.reset_map()
        self.setup_agents()

        observations = []
        
        for i in range(self.num_agents):
            observations.append(self.get_obs_agent(i))
        
        return observations


    def get_obs(self):
        """ Returns all agent observations in a list """
        os=[]
        for i in range(self.num_agents):
            os.append(self.get_obs_agent(i))
        return os


    def get_obs_size(self):
        """ Returns the shape of the observation """
        return len(self.get_obs_agent(0))

    def get_state(self):
        s=self.get_obs()
        s=np.array(s)
        s=np.concatenate(s,axis=0)
        return s

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.get_obs_size()*self.num_agents

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.num_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return np.ones(self.action_space[0].n)

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.action_space[0].n

    def get_stats(self):
        return None

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.num_agents,
                    "episode_limit": self.episode_length}
        return env_info
    

