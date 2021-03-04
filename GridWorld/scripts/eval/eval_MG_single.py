import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from envs import MGSingleEnv
from algorithm.ppo import PPO
from algorithm.model import Policy

from config import get_config
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv, SingleSubprocVecEnv, SingleDummyVecEnv, SingleSubprocVecEnv2, SingleDummyVecEnv2
from utils.util import update_linear_schedule
from utils.storage import RolloutStorage
from utils import utility_funcs
import os
import shutil
import imageio
    
def main():
    args = get_config()
    
    # cuda
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(1)
    else:
        device = torch.device("cpu")
        torch.set_num_threads(args.n_training_threads)
    
    run_dir = Path(args.model_dir) / ('run'+str(args.seed)) / 'eval_adaptive_script'
    if os.path.exists(run_dir): 
        shutil.rmtree(run_dir)
        os.mkdir(run_dir)
    log_dir = run_dir / 'logs'
    os.makedirs(str(log_dir))
    logger = SummaryWriter(str(log_dir))
		
    assert args.num_agents == 1, ("only 1 agent is supported, check the config.py.")	
    env = MGSingleEnv(args)		

    #Policy network     
    # agent 0
    actor_critic = torch.load(str(args.model_dir) + "/run" + str(args.seed) + "/adaptive/models/agent0_model.pt")['model'].to(device)
               
    for policy in range(args.num_opponents): 
        coop_num = []
        defect_num = []
        coopdefect_num = []
        defectcoop_num = []
        collective_return = []
        eval_rewards = [] 
        for episode in range(args.eval_episodes):  
            print("Policy %i of %i" % (policy, args.num_policy_candidates))
            if args.critic_full_obs:
                state, state_critic, select_opponent = env.reset(choose = policy)
                state_critic = np.array([state_critic])	
            else:
                state, select_opponent = env.reset(choose = policy)
            state = np.array([state])	
                    
            policy_reward = 0	

            # rollout		
            if len(env.observation_space[0]) == 1:
                if args.critic_full_obs:	
                    share_obs = (torch.tensor(state_critic.reshape(1, -1),dtype=torch.float32)).to(device)
                else:
                    share_obs = (torch.tensor(state.reshape(1, -1),dtype=torch.float32)).to(device)
                obs = (torch.tensor(state[:,0,:],dtype=torch.float32)).to(device)
            elif len(env.observation_space[0]) == 3:
                if args.critic_full_obs:	
                    share_obs = (torch.tensor(state_critic.reshape(1, -1, env.observation_space[0][1], env.observation_space[0][2]),dtype=torch.float32)).to(device)
                else:
                    share_obs = (torch.tensor(state.reshape(1, -1, env.observation_space[0][1], env.observation_space[0][2]),dtype=torch.float32)).to(device)
                obs = (torch.tensor(state[:,0,:,:,:],dtype=torch.float32)).to(device)
            else:	
                raise NotImplementedError	
            recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size).to(device)
            recurrent_hidden_states_critic = torch.zeros(1, actor_critic.recurrent_hidden_state_size).to(device)
            recurrent_c_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size).to(device)
            recurrent_c_states_critic = torch.zeros(1, actor_critic.recurrent_hidden_state_size).to(device)	
            masks = torch.ones(1,1).to(device)	
                    
            for step in range(args.episode_length):	   
                print("step %i of %i" % (step, args.episode_length))	
                # Sample actions                                      
                one_hot_action = np.zeros((1,env.action_space[0].n))	
                with torch.no_grad():	
                    value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic, recurrent_c_states, recurrent_c_states_critic = actor_critic.act(share_obs, obs, recurrent_hidden_states, recurrent_hidden_states_critic, recurrent_c_states, recurrent_c_states_critic, masks)              	
                one_hot_action[0][action] = 1	
                
                # Obser reward and next obs	
                if args.critic_full_obs:
                    state, state_critic, select_opponent, reward, done, infos = env.step(one_hot_action)
                    state_critic = np.array([state_critic])	
                else:
                    state, select_opponent, reward, done, infos = env.step(one_hot_action)

                for i in range(args.num_agents):	
                    print("Reward of agent%i: " %i + str(reward[i]))	
                    policy_reward += reward[i]

                state = np.array([state])
                    
                if len(env.observation_space[0]) == 1:
                    if args.critic_full_obs:	
                        share_obs = (torch.tensor(state_critic.reshape(1, -1),dtype=torch.float32)).to(device)
                    else:
                        share_obs = (torch.tensor(state.reshape(1, -1),dtype=torch.float32)).to(device)
                    obs = (torch.tensor(state[:,0,:],dtype=torch.float32)).to(device)
                elif len(env.observation_space[0]) == 3:
                    if args.critic_full_obs:	
                        share_obs = (torch.tensor(state_critic.reshape(1, -1, env.observation_space[0][1], env.observation_space[0][2]),dtype=torch.float32)).to(device)
                    else:
                        share_obs = (torch.tensor(state.reshape(1, -1, env.observation_space[0][1], env.observation_space[0][2]),dtype=torch.float32)).to(device)
                    obs = (torch.tensor(state[:,0,:,:,:],dtype=torch.float32)).to(device)
                else:	
                    raise NotImplementedError
                        
            print("Reward of Policy%i: " %(policy+1) + str(policy_reward))
            eval_rewards.append(policy_reward)
            
            if args.env_name == "StagHuntSingle":
                if 'coop&coop_num' in infos.keys():
                    coop_num.append(infos['coop&coop_num'])
                if 'defect&defect_num' in infos.keys():
                    defect_num.append(infos['defect&defect_num'])
                if 'coop&defect_num' in infos.keys():
                    coopdefect_num.append(infos['coop&defect_num'])
                if 'defect&coop_num' in infos.keys():
                    defectcoop_num.append(infos['defect&coop_num'])
                            
                logger.add_scalars("Policy-" + str(select_opponent+1) + '/coop&coop_num_per_episode',
                        {'coop&coop_num_per_episode': coop_num[episode]},
                        episode)
                logger.add_scalars("Policy-" + str(select_opponent+1) + '/defect&defect_num_per_episode',
                        {'defect&defect_num_per_episode': defect_num[episode]},
                        episode)
                logger.add_scalars("Policy-" + str(select_opponent+1) + '/coop&defect_num_per_episode',
                        {'coop&defect_num_per_episode': coopdefect_num[episode]},
                        episode)
                logger.add_scalars("Policy-" + str(select_opponent+1) +'/defect&coop_num_per_episode',
                        {'defect&coop_num_per_episode': defectcoop_num[episode]},
                        episode)
                logger.add_scalars("Policy-" + str(select_opponent+1) +'/collective_return',
                        {'collective_return': eval_rewards[episode]*10},
                        episode)
                    
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()   
    
if __name__ == "__main__":
    main()