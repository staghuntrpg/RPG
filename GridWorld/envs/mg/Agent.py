from gym.spaces import Discrete
import numpy as np

COOP = 0
DEFECT = 1

class Agent(object):

    def __init__(self, history_length):
        """Superclass for all agents.

        Parameters
        ----------

        """
        
        self.action_space = Discrete(2)
        self.observation_space = [history_length * 2]# Discrete(history_length * 2)
        self.state = [-1] * history_length 
        self.reward = 0
        self.done = False
        self.info = None 

class CoopAgent(Agent):
    '''
        @brief: always play coop
    '''

    def __init__(self, history_length):
        super().__init__(history_length)
        self.agent0_action = -1


    def act(self):
        return COOP

class DefectAgent(Agent):
    '''
        @brief: always play defect
    '''

    def __init__(self, history_length):
        super().__init__(history_length)
        self.agent0_action = -1


    def act(self):
        return DEFECT 

class RandomAgent(Agent):
    '''
        @brief: randomly play
    '''

    def __init__(self, history_length):
        super().__init__(history_length)
        self.agent0_action = -1


    def act(self):
        return np.random.randint(0,2)

class TFTAgent(Agent):
    '''
        @brief: randomly play
    '''

    def __init__(self, history_length):
        super().__init__(history_length)
        self.agent0_action = -1
        self.step = 0


    def act(self):
        if self.step == 0:
            action=COOP
        else:
            action = self.agent0_action

        self.step += 1
        return action