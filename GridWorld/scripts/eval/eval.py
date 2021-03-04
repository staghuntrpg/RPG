import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from envs import MGEnv, CleanupEnv, HarvestEnv, GridWorldEnv,GridWorldAdaptiveEnv, MGSingleEnv, MGAdaptiveEnv
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
    
    run_dir = Path(args.model_dir)/ ("run" + str(args.seed)) / 'eval_finetune'
    if os.path.exists(run_dir): 
        shutil.rmtree(run_dir)
        os.mkdir(run_dir)
    log_dir = run_dir / 'logs'
    os.makedirs(str(log_dir))
    logger = SummaryWriter(str(log_dir))
    gifs_dir = run_dir / 'gifs'	
    os.makedirs(str(gifs_dir))
	
    # env	
    if args.env_name == "StagHunt":	
        assert args.num_agents == 2, ("only 2 agents is supported, check the config.py.")	
        env = MGEnv(args)		
    elif args.env_name == "StagHuntGW" or args.env_name == "EscalationGW":	
        assert args.num_agents == 2, ("only 2 agent is supported in single navigation, check the config.py.")	
        env = GridWorldEnv(args)		
    else:	
        print("Can not support the " + args.env_name + "environment." )	
        raise NotImplementedError	

    #Policy network    	    
    actor_critic = []	    
    for i in range(args.num_agents):
        ac = torch.load(str(args.model_dir) + 'run' + str(args.seed) + "/finetune/models/agent%i_model" % i + ".pt")['model'].to(device)
        actor_critic.append(ac)
        
    coop_num = []
    defect_num = []
    coopdefect_num = []
    defectcoop_num = []
    gore1_num = []
    gore2_num = []
    hare1_num = []
    hare2_num = []
    collective_return = []
    apple_consumption = []
    waste_cleared = []
    sustainability = []
    fire = []
    eval_rewards = []
    frames = []
        
    for episode in range(args.eval_episodes):  
        print("Episode %i of %i" % (episode, args.eval_episodes))
        state = env.reset()	
        state = np.array([state])	
        
        share_obs = []	
        obs = []	
        recurrent_hidden_statess = []	
        recurrent_hidden_statess_critic = []	
        recurrent_c_statess = []	
        recurrent_c_statess_critic = []	
        masks = []	
        policy_reward = 0	

        # rollout	
        for i in range(args.num_agents):	
            if len(env.observation_space[0]) == 1:	
                share_obs.append((torch.tensor(state.reshape(1, -1),dtype=torch.float32)).to(device))	
                obs.append((torch.tensor(state[:,i,:],dtype=torch.float32)).to(device))	
            else:	
                raise NotImplementedError	
            recurrent_hidden_statess.append(torch.zeros(1, actor_critic[i].recurrent_hidden_state_size).to(device))	
            recurrent_hidden_statess_critic.append(torch.zeros(1, actor_critic[i].recurrent_hidden_state_size).to(device))	
            recurrent_c_statess.append(torch.zeros(1, actor_critic[i].recurrent_hidden_state_size).to(device))	
            recurrent_c_statess_critic.append(torch.zeros(1, actor_critic[i].recurrent_hidden_state_size).to(device))	
            masks.append(torch.ones(1,1).to(device))	
            
        frames_dir = str(gifs_dir) + '/episode%i/'%episode + 'frames/'
        for step in range(args.episode_length):	   
            print("step %i of %i" % (step, args.episode_length))	
            # Sample actions
            
            if not os.path.exists(frames_dir):
                os.makedirs(frames_dir)
                
            if args.save_gifs:
                frame = env.render(filename= frames_dir + str(step).zfill(6) + '.png')
                frames.append(frame)	
            	
            one_hot_actions = []	
            for i in range(args.num_agents):
                one_hot_action = np.zeros(env.action_space[0].n)	
                with torch.no_grad():	
                    value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic,recurrent_c_states, recurrent_c_states_critic = actor_critic[i].act(share_obs[i], obs[i], recurrent_hidden_statess[i], recurrent_hidden_statess_critic[i], recurrent_c_statess[i], recurrent_c_statess_critic[i], masks[i])	
                recurrent_hidden_statess[i].copy_(recurrent_hidden_states)	
                recurrent_hidden_statess_critic[i].copy_(recurrent_hidden_states_critic) 	
                recurrent_c_statess[i].copy_(recurrent_c_states)	
                recurrent_c_statess_critic[i].copy_(recurrent_c_states_critic)              	
                one_hot_action[action] = 1	
                one_hot_actions.append(one_hot_action)	

            # Obser reward and next obs	
            state, reward, done, infos = env.step(one_hot_actions)
            if any(done):
                break	

            for i in range(args.num_agents):	
                print("Reward of agent%i: " %i + str(reward[i]))	
                policy_reward += reward[i]	
            state = np.array([state])	

            for i in range(args.num_agents):	
                if len(env.observation_space[0]) == 1:	
                    share_obs[i].copy_(torch.tensor(state.reshape(1, -1),dtype=torch.float32))	
                    obs[i].copy_(torch.tensor(state[:,i,:],dtype=torch.float32))	
                 
        eval_rewards.append(policy_reward)
        if args.save_gifs:
            utility_funcs.make_gif_from_image_dir(str(gifs_dir) + '/episode%i/'%episode, frames_dir, gif_name=args.env_name + '_trajectory')
        if args.env_name == "StagHunt":
            if 'coop&coop_num' in infos.keys():
                coop_num.append(infos['coop&coop_num'])
            if 'defect&defect_num' in infos.keys():
                defect_num.append(infos['defect&defect_num'])
            if 'coop&defect_num' in infos.keys():
                coopdefect_num.append(infos['coop&defect_num'])
            if 'defect&coop_num' in infos.keys():
                defectcoop_num.append(infos['defect&coop_num'])
                        
            logger.add_scalars('coop&coop_num_per_episode',
                    {'coop&coop_num_per_episode': coop_num[episode]},
                    episode)
            logger.add_scalars('defect&defect_num_per_episode',
                    {'defect&defect_num_per_episode': defect_num[episode]},
                    episode)
            logger.add_scalars('coop&defect_num_per_episode',
                    {'coop&defect_num_per_episode': coopdefect_num[episode]},
                    episode)
            logger.add_scalars('defect&coop_num_per_episode',
                    {'defect&coop_num_per_episode': defectcoop_num[episode]},
                    episode)
            logger.add_scalars('collective_return',
                    {'collective_return': eval_rewards[episode]*10},
                    episode)
        elif args.env_name == "StagHuntGW":
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

            logger.add_scalars('collective_return',
                {'collective_return': collective_return[episode]},
                episode)
            logger.add_scalars('coop&coop_num_per_episode',
                    {'coop&coop_num_per_episode': coop_num[episode]},
                    episode)
            logger.add_scalars('gore1_num_per_episode',
                    {'gore1_num_per_episode': gore1_num[episode]},
                    episode)
            logger.add_scalars('gore2_num_per_episode',
                    {'gore2_num_per_episode': gore2_num[episode]},
                    episode)
            logger.add_scalars('hare1_num_per_episode',
                    {'hare1_num_per_episode': hare1_num[episode]},
                    episode)
            logger.add_scalars('hare2_num_per_episode',
                    {'hare2_num_per_episode': hare2_num[episode]},
                    episode)
                    
        elif args.env_name == "EscalationGW":
            if 'collective_return' in infos.keys(): 
                collective_return.append(infos['collective_return']) 
            if 'coop&coop_num' in infos.keys():
                coop_num.append(infos['coop&coop_num'])

            logger.add_scalars('collective_return',
                {'collective_return': collective_return[episode]},
                episode)
            logger.add_scalars('coop&coop_num_per_episode',
                    {'coop&coop_num_per_episode': coop_num[episode]},
                    episode)
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()   
    r = np.mean(np.array(eval_rewards))
    print("Mean reward is %i" % r)
    
if __name__ == "__main__":
    main()