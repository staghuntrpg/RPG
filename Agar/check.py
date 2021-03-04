import glob
import os
import time
from collections import deque
import random
from gv import *

import numpy as np
from datetime import datetime

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage

def evaluate(args_, agent, device, out_file, out_bool = True, eval = None):

    args = deepcopy(args_)
    set_v('total_step', 0)
    args.eval = eval
    envs = make_vec_envs(args, args.seed, args.num_processes, args.gamma, device)
    
    n_agent = args.num_controlled_agent

    rollouts = [RolloutStorage(args.num_steps, args.num_processes, get_v('obs_size'), 3, agent[0].actor_critic.recurrent_hidden_state_size) for i in range(n_agent)]

    obs = envs.reset()
    for i in range(n_agent):
        rollouts[i].obs[0].copy_(obs['t'+str(i)])
        rollouts[i].to(device)

    episode_rewards = [deque(maxlen=2048) for i in range(2)]
    episode_rewards_g = [deque(maxlen=2048) for i in range(2)]
    hit = [[], []]
    dis = [[], []]
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    value = [[] for i in range(n_agent)]
    action = [[] for i in range(n_agent)]
    action_log_prob = [[] for i in range(n_agent)]
    recurrent_hidden_states = [[] for i in range(n_agent)]

    for j in range(num_updates):
        for step in range(args.num_steps):
            
            with torch.no_grad():
                for i in range(n_agent):
                    value[i], action[i], action_log_prob[i], recurrent_hidden_states[i] = agent[0].actor_critic.act(
                        rollouts[i].obs[step], rollouts[i].recurrent_hidden_states[step],
                        rollouts[i].masks[step], deterministic = True)
            # Obser reward and next obs:
            obs, reward, done, infos = envs.step(torch.cat(action, -1).cpu())
            out = 'info in eva3 ' + str(torch.cat(action, -1)[0].cpu()) + '   ' + str(reward[0])
            if out_bool:
                print(out)
                out_file.write(out)
            
            for i in range(args.num_processes):
                k = 0
                for info in infos[i]:
                    if 'episode' in info.keys():
                        episode_rewards[k].append(info['episode']['r'])
                        episode_rewards_g[k].append(info['episode']['r_g'])
                        hit[k].append(info['episode']['hit'])
                        dis[k].append(info['episode']['dis'])
                        if len(hit[0]) + len(hit[1]) >= 2 * 64:
                            
                            for q in range(args.num_controlled_agent):
                                hit[q] = np.array(hit[q]).mean(0)
                                dis[q] = np.array(dis[q]).mean(0)
                                print('std', np.std(episode_rewards[q]), np.std(episode_rewards_g[q]))
                                #print(episode_rewards_g)
                                episode_rewards[q] = np.mean(episode_rewards[q])
                                episode_rewards_g[q] = np.mean(episode_rewards_g[q])
                            envs.close()
                            print('evaluation:', np.mean(hit, 0), np.mean(episode_rewards), np.mean(episode_rewards_g))
                            return hit, dis, episode_rewards, episode_rewards_g
                    k += 1

            # If done then clean the history of observations.
            for i in range(n_agent):
                masks = torch.FloatTensor(
                    [[0.0] if done[k][i] else [1.0] for k in range(args.num_processes)])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in infos[k][i].keys() else [1.0]
                     for k in range(args.num_processes)])
                rollouts[i].insert(obs['t'+str(i)], recurrent_hidden_states[i], action[i],
                                action_log_prob[i], value[i], reward[:,i,:], masks, bad_masks)

        for i in range(n_agent):
            rollouts[i].after_update()

if __name__ == "__main__":
    main()
