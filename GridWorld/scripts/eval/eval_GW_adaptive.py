import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from envs import GridWorldAdaptiveEnv
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
    if args.save_gifs:
        gifs_dir = run_dir / 'gifs'	
        os.makedirs(str(gifs_dir))
    
    policy_candidates = []
    for i in range(args.num_policy_candidates-2):
        ac = torch.load(str(args.model_dir) + "/run" + str(args.seed) +"/models/Policy" + str(i+5) + "-agent0_model.pt")['model'].cpu()
        print(i)
        policy_candidates.append(ac)
		
    assert args.num_agents == 2, ("only 2 agent is supported, check the config.py.")	
    env = GridWorldAdaptiveEnv(args, policy_candidates)		

    #Policy network     
    # agent 0
    actor_critic = torch.load("/home/yuchao/project/paper/run" + str(args.seed) + "/adaptive/models/agent0_model.pt")['model'].to(device)
    if args.save_gifs:
        frames_dir = str(gifs_dir) + '/frames/'
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir)
        
    for policy in range(args.num_policy_candidates-2): 
        coop_num = []
        gore1_num = []
        gore2_num = []
        hare1_num = []
        hare2_num = []
        collective_return = []
        eval_rewards = [] 
        for episode in range(args.eval_episodes):  
            print("Policy %i of %i" % (policy+1, args.num_policy_candidates-2))
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
                if args.save_gifs:
                    env.render(filename = frames_dir + str(step + policy * args.episode_length).zfill(6) + '.png')	
                        
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
                else:	
                    raise NotImplementedError
                        
            print("Reward of Policy%i: " %(policy+1) + str(policy_reward))
            
            if args.env_name == "StagHuntGWAdaptive":
                if 'collective_return' in infos.keys(): 
                    collective_return.append(infos['collective_return']) 
                if 'coop&coop_num' in infos.keys():
                    coop_num.append(infos['coop&coop_num'])
                if 'gore1_num' in infos.keys(): 
                    gore1_num.append(infos['gore1_num']) 
                if 'gore2_num' in infos.keys():
                    gore2_num.append(infos['gore2_num'])
                if 'hare1_num' in infos.keys(): 
                    hare1_num.append(infos['hare1_num']) 
                if 'hare2_num' in infos.keys():
                    hare2_num.append(infos['hare2_num'])

                logger.add_scalars('Policy-' + str(select_opponent+1) +'/collective_return',
                    {'collective_return': collective_return[episode]},
                    episode)
                logger.add_scalars('Policy-' + str(select_opponent+1) + '/coop&coop_num_per_episode',
                        {'coop&coop_num_per_episode': coop_num[episode]},
                        episode)
                logger.add_scalars('Policy-' + str(select_opponent+1) + '/gore1_num_per_episode',
                        {'gore1_num_per_episode': gore1_num[episode]},
                        episode)
                logger.add_scalars('Policy-' + str(select_opponent+1) + '/gore2_num_per_episode',
                        {'gore2_num_per_episode': gore2_num[episode]},
                        episode)
                logger.add_scalars('Policy-' + str(select_opponent+1) + '/hare1_num_per_episode',
                        {'hare1_num_per_episode': hare1_num[episode]},
                        episode)
                logger.add_scalars('Policy-' + str(select_opponent+1) + '/hare2_num_per_episode',
                        {'hare2_num_per_episode': hare2_num[episode]},
                        episode)
    if args.save_gifs:                   
        utility_funcs.make_gif_from_image_dir(str(gifs_dir), frames_dir, gif_name=args.env_name + '_trajectory')
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()   
    
if __name__ == "__main__":
    main()