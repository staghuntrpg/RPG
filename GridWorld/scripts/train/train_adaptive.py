#!/usr/bin/env python

import copy
import glob
import os
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from envs import GridWorldAdaptiveEnv, MGAdaptiveEnv
from algorithm.ppo import PPO
from algorithm.model import Policy

from config import get_config
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv, SingleSubprocVecEnv, SingleDummyVecEnv, SingleSubprocVecEnv2, SingleDummyVecEnv2
from utils.util import update_linear_schedule
from utils.storage import RolloutStorage
import shutil

def make_parallel_env(args, policy_candidates=None):
    def get_env_fn(rank):
        def init_env():
            if args.env_name == "StagHuntAdaptive":
                assert args.num_agents == 1, ("only 1 agents is supported, check the config.py.")
                env = MGAdaptiveEnv(args, policy_candidates)
            elif args.env_name == "StagHuntGWAdaptive":
                assert args.num_agents == 2, ("only 2 agents is supported, check the config.py.")
                env = GridWorldAdaptiveEnv(args, policy_candidates)
            else:
                print("Can not support the " + args.env_name + "environment." )
                raise NotImplementedError
            env.seed(args.seed + rank * 1000)
            return env
        return init_env
    if args.n_rollout_threads == 1:
        if args.critic_full_obs:
            return SingleDummyVecEnv2([get_env_fn(0)])
        else:
            return SingleDummyVecEnv([get_env_fn(0)])
    else:
        if args.critic_full_obs:
            return SingleSubprocVecEnv2([get_env_fn(i) for i in range(args.n_rollout_threads)])
        else:
            return SingleSubprocVecEnv([get_env_fn(i) for i in range(args.n_rollout_threads)])
        
def main():
    args = get_config()
    
    # cuda
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(args.n_training_threads)
        if args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
        torch.set_num_threads(args.n_training_threads)
    	
    np.random.seed(args.seed)	
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  
    
    # path
    model_dir = Path('./results') / args.env_name / args.algorithm_name / ("run"+str(args.seed))
    if args.critic_full_obs:
        run_dir = model_dir / 'adaptive'
    else:
        run_dir = model_dir / 'adaptive_only'
    log_dir = run_dir / 'logs'
    save_dir = run_dir / 'models'
    os.makedirs(str(log_dir))
    os.makedirs(str(save_dir))
    logger = SummaryWriter(str(log_dir))  
   
    print("\n Now we have %i fixed policy! Train Single Adaptive Policy... \n" %args.num_policy_candidates)
    args.env_name = args.env_name + "Adaptive" 
    policy_candidates = []
    for i in range(args.num_policy_candidates):
        ac = torch.load(str(model_dir) +("/models/Policy%i" %(i+1)) + "-agent0_model.pt")['model'].cpu()
        policy_candidates.append(ac)
        
    # env
    envs = make_parallel_env(args, policy_candidates)
    
    #Policy network
    # agent 0
    actor_critic = Policy(envs.observation_space[0], 
                    envs.action_space[0],
                    num_agents = args.num_agents,
                    base_kwargs={'lstm': args.lstm,
                                'naive_recurrent': args.naive_recurrent_policy,
                                'recurrent': args.recurrent_policy,
                                'hidden_size': args.hidden_size})
    
    actor_critic.to(device)
    agent0 = PPO(actor_critic,
                0,
                args.clip_param,
                args.ppo_epoch,
                args.num_mini_batch,
                args.data_chunk_length,
                args.value_loss_coef,
                args.entropy_coef,
                logger,
                lr=args.lr,
                eps=args.eps,
                max_grad_norm=args.max_grad_norm,
                use_clipped_value_loss= args.use_clipped_value_loss)
    #replay buffer
    rollout = RolloutStorage(args.num_agents,
                        0,
                        args.episode_length, 
                        args.n_rollout_threads,
                        envs.observation_space[0], 
                        envs.action_space[0],
                        actor_critic.recurrent_hidden_state_size)
    
    # reset
    if args.critic_full_obs:
        obs, obs_critic, select_opponent = envs.reset()
    else:
        obs, select_opponent = envs.reset()
            
    # rollout
    if len(envs.observation_space[0]) == 1:
        if args.critic_full_obs:
            rollout.share_obs[0].copy_(torch.tensor(obs_critic.reshape(args.n_rollout_threads, -1)))
        else:
            rollout.share_obs[0].copy_(torch.tensor(obs.reshape(args.n_rollout_threads, -1)))
        rollout.obs[0].copy_(torch.tensor(obs[:,0,:]))
        rollout.recurrent_hidden_states.zero_()
        rollout.recurrent_hidden_states_critic.zero_()
        rollout.recurrent_c_states.zero_()
        rollout.recurrent_c_states_critic.zero_()
    else:
        raise NotImplementedError
    rollout.to(device) 

    # run
    collective_return = []
    apple_consumption = []
    waste_cleared = []
    sustainability = []
    fire = []

    start = time.time()
    episodes = int(args.num_env_steps) // args.episode_length // args.n_rollout_threads
    all_episode = 0
    all_episode_adaptive = np.zeros(args.num_policy_candidates)

    for episode in range(episodes):
        if args.use_linear_lr_decay:
            update_linear_schedule(agent0.optimizer, 
                                    episode, 
                                    episodes, 
                                    args.lr)

        for step in range(args.episode_length):
            with torch.no_grad():
                value, action0, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic, recurrent_c_states, recurrent_c_states_critic =                                            actor_critic.act(rollout.share_obs[step], 
                                                    rollout.obs[step], 
                                                    rollout.recurrent_hidden_states[step], 
                                                    rollout.recurrent_hidden_states_critic[step], 
                                                    rollout.recurrent_c_states[step], 
                                                    rollout.recurrent_c_states_critic[step],
                                                    rollout.masks[step])
            
            # rearrange action           
            actions_env = []
            for i in range(args.n_rollout_threads):
                one_hot_action = np.zeros((1, envs.action_space[0].n))
                one_hot_action[0][action0[i]] = 1
                actions_env.append(one_hot_action)
            
            
            # Obser reward and next obs
            if args.critic_full_obs:
                obs, obs_critic, select_opponent, reward, done, infos = envs.step(actions_env)
            else:
                obs, select_opponent, reward, done, infos = envs.step(actions_env)

            # If done then clean the history of observations.
            # insert data in buffer    
            masks = []
            bad_masks = []
            for i in range(args.num_agents):
                mask = []
                bad_mask = []
                for done_ in done:  
                    if done_[i]:              
                        mask.append([0.0])
                        bad_mask.append([1.0])
                    else:
                        mask.append([1.0])
                        bad_mask.append([1.0])
                masks.append(torch.FloatTensor(mask))
                bad_masks.append(torch.FloatTensor(bad_mask))

            if len(envs.observation_space[0]) == 1:
                if args.critic_full_obs:
                    rollout.insert(torch.tensor(obs_critic.reshape(args.n_rollout_threads, -1)), 
                                    torch.tensor(obs[:,0,:]), 
                                    recurrent_hidden_states, 
                                    recurrent_hidden_states_critic, 
                                    recurrent_c_states, 
                                    recurrent_c_states_critic, 
                                    action0,
                                    action_log_prob, 
                                    value, 
                                    torch.tensor(reward[:, 0].reshape(-1,1)), 
                                    masks[0], 
                                    bad_masks[0])
                else:
                    rollout.insert(torch.tensor(obs.reshape(args.n_rollout_threads, -1)), 
                                    torch.tensor(obs[:,0,:]), 
                                    recurrent_hidden_states, 
                                    recurrent_hidden_states_critic,
                                    recurrent_c_states, 
                                    recurrent_c_states_critic, 
                                    action0,
                                    action_log_prob, 
                                    value, 
                                    torch.tensor(reward[:, 0].reshape(-1,1)), 
                                    masks[0], 
                                    bad_masks[0])
            else:
                raise NotImplementedError

        with torch.no_grad():
            next_value = actor_critic.get_value(rollout.share_obs[-1], 
                                                rollout.obs[-1], 
                                                rollout.recurrent_hidden_states[-1],
                                                rollout.recurrent_hidden_states_critic[-1],
                                                rollout.recurrent_c_states[-1],
                                                rollout.recurrent_c_states_critic[-1],
                                                rollout.masks[-1]).detach()

        rollout.compute_returns(next_value, 
                                args.use_gae, 
                                args.gamma,
                                args.gae_lambda, 
                                args.use_proper_time_limits)

        # update the network
        value_loss, action_loss, dist_entropy = agent0.update(rollout)
                
        if args.env_name == "StagHuntAdaptive":
            coop_num = []
            defect_num = []
            coopdefect_num = []
            defectcoop_num = []
            rew = []
            for info in infos:
                if 'coop&coop_num' in info.keys():
                    coop_num.append(info['coop&coop_num'])
                if 'defect&defect_num' in info.keys():
                    defect_num.append(info['defect&defect_num'])
                if 'coop&defect_num' in info.keys():
                    coopdefect_num.append(info['coop&defect_num'])
                if 'defect&coop_num' in info.keys():
                    defectcoop_num.append(info['defect&coop_num'])
            for i in range(args.n_rollout_threads):
                rew.append(rollout.rewards[:,i,:].sum().cpu().numpy())
            
            for i in range(args.n_rollout_threads):  
                logger.add_scalars('Policy-' + str(select_opponent[i]+1) + '/coop&coop_num_per_episode',
                        {'coop&coop_num_per_episode': coop_num[i]},
                        all_episode_adaptive[select_opponent[i]])
                logger.add_scalars('Policy-' + str(select_opponent[i]+1) + '/defect&defect_num_per_episode',
                        {'defect&defect_num_per_episode': defect_num[i]},
                        all_episode_adaptive[select_opponent[i]])
                logger.add_scalars('Policy-' + str(select_opponent[i]+1) + '/coop&defect_num_per_episode',
                        {'coop&defect_num_per_episode': coopdefect_num[i]},
                        all_episode_adaptive[select_opponent[i]])
                logger.add_scalars('Policy-' + str(select_opponent[i]+1) + '/defect&coop_num_per_episode',
                        {'defect&coop_num_per_episode': defectcoop_num[i]},
                        all_episode_adaptive[select_opponent[i]])
                logger.add_scalars('Policy-' + str(select_opponent[i]+1) + '/reward',
                        {'reward': np.mean(np.array(rew[i]))},
                        all_episode_adaptive[select_opponent[i]])                    
                all_episode_adaptive[select_opponent[i]] += 1
        elif args.env_name == "StagHuntGWAdaptive":
            collective_return = []
            coop_num = []
            gore1_num = []
            gore2_num = []
            hare1_num = []
            hare2_num = []
            for info in infos:
                if 'collective_return' in info.keys(): 
                    collective_return.append(info['collective_return']) 
                if 'coop&coop_num' in info.keys():
                    coop_num.append(info['coop&coop_num'])
                if 'gore1_num' in info.keys(): 
                    gore1_num.append(info['gore1_num'])
                if 'gore2_num' in info.keys(): 
                    gore2_num.append(info['gore2_num'])
                if 'hare1_num' in info.keys(): 
                    hare1_num.append(info['hare1_num'])
                if 'hare2_num' in info.keys(): 
                    hare2_num.append(info['hare2_num'])

            for i in range(args.n_rollout_threads):  
                logger.add_scalars('Policy-' + str(select_opponent[i]+1) + '/collective_return',
                    {'collective_return': collective_return[i]},
                    all_episode_adaptive[select_opponent[i]])
                logger.add_scalars('Policy-' + str(select_opponent[i]+1) + '/coop&coop_num_per_episode',
                        {'coop&coop_num_per_episode': coop_num[i]},
                        all_episode_adaptive[select_opponent[i]])
                logger.add_scalars('Policy-' + str(select_opponent[i]+1) + '/gore1_num_per_episode',
                    {'gore1_num_per_episode': gore1_num[i]},
                    all_episode_adaptive[select_opponent[i]]) 
                logger.add_scalars('Policy-' + str(select_opponent[i]+1) + '/gore2_num_per_episode',
                    {'gore2_num_per_episode': gore2_num[i]},
                    all_episode_adaptive[select_opponent[i]])
                logger.add_scalars('Policy-' + str(select_opponent[i]+1) + '/hare1_num_per_episode',
                    {'hare1_num_per_episode': hare1_num[i]},
                    all_episode_adaptive[select_opponent[i]]) 
                logger.add_scalars('Policy-' + str(select_opponent[i]+1) + '/hare2_num_per_episode',
                    {'hare2_num_per_episode': hare2_num[i]},
                    all_episode_adaptive[select_opponent[i]])   
                all_episode_adaptive[select_opponent[i]] += 1 

        if args.critic_full_obs:
            obs, obs_critic, select_opponent = envs.reset()
        else:
            obs, select_opponent = envs.reset()

        if len(envs.observation_space[0]) == 1:
            if args.critic_full_obs:
                rollout.share_obs[0].copy_(torch.tensor(obs_critic.reshape(args.n_rollout_threads, -1)))
            else:
                rollout.share_obs[0].copy_(torch.tensor(obs.reshape(args.n_rollout_threads, -1)))
            rollout.obs[0].copy_(torch.tensor(obs[:,0,:]))
            rollout.recurrent_hidden_states.zero_()
            rollout.recurrent_hidden_states_critic.zero_()
            rollout.recurrent_c_states.zero_()
            rollout.recurrent_c_states_critic.zero_()
            rollout.masks[0].copy_(torch.ones(args.n_rollout_threads, 1))
            rollout.bad_masks[0].copy_(torch.ones(args.n_rollout_threads, 1))
        else:
            raise NotImplementedError
        rollout.to(device)

        if (episode % args.save_interval == 0 or episode == episodes - 1):            
            torch.save({
                    'model': actor_critic
                    }, 
                    str(save_dir) + "/agent0_model.pt")

        # log information
        if episode % args.log_interval == 0:
            total_num_steps = (episode + 1) * args.episode_length * args.n_rollout_threads
            end = time.time()
            print("\n Updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                .format(episode, 
                        episodes,
                        total_num_steps,
                        args.num_env_steps,
                        int(total_num_steps / (end - start))))
            print("value loss: agent0--" + str(value_loss))
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
       
if __name__ == "__main__":
    main()
