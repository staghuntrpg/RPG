from copy import deepcopy
import glob
import os
import time
from collections import deque
from gv import *
gv_init()

import numpy as np
from datetime import datetime
time_str = str(datetime.now()).replace(':','-').replace(' ','--')
time_str = time_str[0:time_str.find(".")]
set_v('time_str', time_str)
set_v('obs_size', 578)

def get_best_gpu(force = None):
    if force is not None:return force
    s = os.popen("nvidia-smi --query-gpu=memory.free --format=csv")
    a = []
    ss = s.read().replace('MiB','').replace('memory.free','').split('\n')
    s.close()
    for i in range(1, len(ss) - 1):
        a.append(int(ss[i]))
    print(a)
    best = int(np.argmax(a))
    print('the best GPU is ',best,' with free memories of ',ss[best + 1])
    return best

def get_name():

    s = os.popen("whoami")
    a = []
    ss = s.read().replace('\n', '')
    s.close()
    print('user name is', ss)
    return ss

gpu_id = get_best_gpu()
user_name = get_name()

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.cuda.set_device(gpu_id)

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate


def main():

    set_v('total_step', 0)
    args = get_args() 
    his_r = []
    his_r_t = []
    his_r_g = [[], []]
    his_r_g_t = []
    his_r_g_t_i = []
    his_hit = [[[], [], [], []], [[], [], [], []]]
    his_hit_t = [[], [], [], []]
    his_dis, his_dis_t = [], []
    his_n_step = []
    his_all_n_step = []
    his_ent    = []
    his_c_ent = []
    his_d_ent = []
    his_norm = []
    his_g_norm = []
    his_vl = []
    now_save_time = -1
    args.seed = int(1000000 * np.random.random_sample())
    #torch.manual_seed(args.seed)
    #torch.cuda.manual_seed_all(args.seed)
    
    load = False
    if args.load_time_label is not None:
        load_path = os.path.join('trained_models', args.load_time_label)
        his_ent=np.load(os.path.join(load_path, 'his_ent.npy'), allow_pickle = True).tolist()
        his_g_norm=np.load(os.path.join(load_path, 'his_g_norm.npy'), allow_pickle = True).tolist()
        his_norm=np.load(os.path.join(load_path, 'his_norm.npy'), allow_pickle = True).tolist()
        his_n_step=np.load(os.path.join(load_path, 'his_n_step.npy'), allow_pickle = True).tolist()
        his_c_ent=np.load(os.path.join(load_path, 'his_c_ent.npy'), allow_pickle = True).tolist()
        his_d_ent=np.load(os.path.join(load_path, 'his_d_ent.npy'), allow_pickle = True).tolist()
        his_vl=np.load(os.path.join(load_path, 'his_vl.npy'), allow_pickle = True).tolist()
        
        his_r=np.load(os.path.join(load_path, 'his_r.npy'), allow_pickle = True).tolist()
        his_r_g=np.load(os.path.join(load_path, 'his_r_g.npy'), allow_pickle = True).tolist()
        his_hit=np.load(os.path.join(load_path, 'his_hit.npy'), allow_pickle = True).tolist()
        his_dis=np.load(os.path.join(load_path, 'his_dis.npy'), allow_pickle = True).tolist()
        his_r_t=np.load(os.path.join(load_path, 'his_r_t.npy'), allow_pickle = True).tolist()
        his_r_g_t=np.load(os.path.join(load_path, 'his_r_g_t.npy'), allow_pickle = True).tolist()
        his_r_g_t_i=np.load(os.path.join(load_path, 'his_r_g_t_i.npy'), allow_pickle = True).tolist()
        his_hit_t=np.load(os.path.join(load_path, 'his_hit_t.npy'), allow_pickle = True).tolist()
        his_dis_t=np.load(os.path.join(load_path, 'his_dis_t.npy'), allow_pickle = True).tolist()
        

        args=np.load(os.path.join(load_path, 'args.npy'), allow_pickle = True)[0]
        load = True
    args.eval = None

    set_v('n_agent', args.num_controlled_agent)
    
    save_path = os.path.join(args.save_dir, time_str)
    os.makedirs(save_path)
    np.save(os.path.join(save_path, "args.npy"), np.array([args]))
    
    out_file = open(os.path.join(save_path, "out.txt"),"a+")
    out_file.write(str(args))
    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True #because of strange reason, without these 2 flags, the NN may run a little different (for faster speed)

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(8)
    device = torch.device("cuda" if args.cuda else "cpu")
    set_v('device', device)
    
    total_num_steps = 0
    args.total_step = 0
    if load:
        total_num_steps = his_n_step[-1]
        args.total_step = total_num_steps
        now_save_time = total_num_steps // 1000000
        print('real total steps', total_num_steps)
        his_all_n_step = [i * total_num_steps / len(his_r_t) for i in range(len(his_r_t))]
    
    envs = make_vec_envs(args, args.seed, args.num_processes,args.gamma, device)
    
    n_agent = args.num_controlled_agent

    if load:
        agent = torch.load(os.path.join(load_path, 'agent.pt'), map_location = 'cuda:'+str(gpu_id))
    else:
        actor_critic = Policy(envs.action_space, base_kwargs={'recurrent': args.recurrent_policy, 'attention': args.num_attention_heads}).to(device)
        agent = [algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm) for i in range(1)]

    rollouts = [RolloutStorage(args.num_steps, args.num_processes, get_v('obs_size'), 3, agent[0].actor_critic.recurrent_hidden_state_size) for i in range(n_agent)]

    obs = envs.reset()
    for i in range(n_agent):
        rollouts[i].obs[0].copy_(obs['t'+str(i)])
        rollouts[i].to(device)

    episode_rewards = deque(maxlen=args.num_processes)
    episode_rewards_g = deque(maxlen=args.num_processes)
    episode_rewards_g_i = deque(maxlen=args.num_processes)
    dis = deque(maxlen=args.num_processes)
    hit = [deque(maxlen=args.num_processes) for i in range(4)]
    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    value = [[] for i in range(n_agent)]
    action = [[] for i in range(n_agent)]
    action_log_prob = [[] for i in range(n_agent)]
    recurrent_hidden_states = [[] for i in range(n_agent)]
    

    for j in range(num_updates):
        print('time:', time.time() - start, 'total_num_steps', total_num_steps)
        set_v('total_step',total_num_steps) 
        for step in range(args.num_steps):
            
            with torch.no_grad():
                for i in range(n_agent):
                    value[i], action[i], action_log_prob[i], recurrent_hidden_states[i] = agent[0].actor_critic.act(
                        rollouts[i].obs[step], rollouts[i].recurrent_hidden_states[step],
                        rollouts[i].masks[step])
            # Obser reward and next obs:
            obs, reward, done, infos = envs.step(torch.cat(action, -1).cpu())
            
            for i in range(args.num_processes):
                for info in infos[i]:
                    if 'episode' in info.keys():
                        episode_rewards.append(info['episode']['r'])
                        episode_rewards_g.append(info['episode']['r_g'])
                        episode_rewards_g_i.append(info['episode']['r_g_i'])
                        dis.append(info['episode']['dis'])
                        for ii in range(4):
                            hit[ii].append(info['episode']['hit'][ii])
                        

            # If done then clean the history of observations.
            for i in range(n_agent):
                masks = torch.FloatTensor(
                    [[0.0] if done[k][i] else [1.0] for k in range(args.num_processes)])
                bad_masks = torch.FloatTensor(
                    [[0.0] if infos[k][i]['bad_transition'] else [1.0]
                     for k in range(args.num_processes)])
                high_masks = torch.Tensor([infos[k][i]['high_masks'] for k in range(args.num_processes)]).to(device).float()
                rollouts[i].insert(obs['t'+str(i)], recurrent_hidden_states[i], action[i], action_log_prob[i], value[i], reward[:,i,:], masks, bad_masks, high_masks = high_masks.reshape(-1, 1))
        for i in range(n_agent):
            with torch.no_grad():
                next_value = agent[0].actor_critic.get_value(
                    rollouts[i].obs[-1], rollouts[i].recurrent_hidden_states[-1],
                    rollouts[i].masks[-1]).detach()

            rollouts[i].compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)
        
        value_loss = []
        action_loss = []
        dist_entropy = []
        total_num_steps += args.num_processes *args.num_steps 
        for i in range(n_agent):
            a, b, c = agent[0].update(rollouts[i], high = True)
            value_loss.append(a)
            action_loss.append(b)
            dist_entropy.append(c)
            rollouts[i].after_update()

        # save for every interval-th episode or for the last epoch
         
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            if total_num_steps // 10000000 > now_save_time:
                now_save_time = total_num_steps // 10000000
                torch.save(agent, os.path.join(save_path, "agent"+str(now_save_time)+".pt"))

            torch.save(agent, os.path.join(save_path, "agent.pt"))
            
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            output = "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.3f}, value_loss {:.5f}, action_loss {:.5f}, norm: {:f}, grad_norm: {:f}, norm2: {:f}, grad_norm2: {:f}\n".format(j, total_num_steps,
                    int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), np.mean(dist_entropy[-1]), np.mean(value_loss[-1]),
                        np.mean(action_loss[-1]), get_v('norm'), get_v('grad_norm'), get_v('norm2'), get_v('grad_norm2'))
            print(output)
            out_file.write(output)
            out_file.flush()
        his_all_n_step.append(total_num_steps)
        his_vl.append(np.mean(value_loss[-1]))
        his_r_t.append(np.mean(episode_rewards))
        his_r_g_t.append(np.mean(episode_rewards_g))
        his_r_g_t_i.append(np.mean(episode_rewards_g_i))
        his_dis_t.append(np.mean(dis))
        for i in range(4):
            his_hit_t[i].append(np.mean(hit[i]))
        np.save(os.path.join(save_path, "his_r_t.npy"), np.array(his_r_t))
        np.save(os.path.join(save_path, "his_r_g_t.npy"), np.array(his_r_g_t))
        np.save(os.path.join(save_path, "his_r_g_t_i.npy"), np.array(his_r_g_t_i))
        np.save(os.path.join(save_path, "his_hit_t.npy"), np.array(his_hit_t))
        np.save(os.path.join(save_path, "his_dis_t.npy"), np.array(his_dis_t))
        np.save(os.path.join(save_path, "his_vl.npy"), np.array(his_vl))
        draw(his_all_n_step, [his_r_t, his_r_g_t], 'n of steps', 'rewards', "rewards during training in 1 episode", save_path, ['γ=1', 'γ=0.99'])
        draw(his_all_n_step, [his_r_g_t_i], 'n of steps', 'rewards', "rewards during training in 1 episode (ind)", save_path, ['γ=0.99'])
        draw(his_all_n_step, his_hit_t, 'n of steps', 'rate', "events' rate during training", save_path, label=['split', 'catch', 'agent', 'cooperate'])
        draw(his_all_n_step, [his_dis_t], 'n of steps', 'dis', "average dis between agents during training", save_path)
        draw(his_all_n_step, [his_vl], 'n of steps', 'v_loss', "average v loss during training", save_path)
        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            t_envs = utils.get_vec_normalize(envs)
            ob_rms = t_envs.ob_rms if t_envs is not None else None
            his_n_step.append(total_num_steps)
            t_hit, t_dis, t_epi_r, t_epi_r_g = [None, None], [None, None], [None, None], [None, None]
            for k in range(2):
                if k == 0:t_hit[k], t_dis[k], t_epi_r[k], t_epi_r_g[k] = evaluate(args, agent, device, out_file, (j // args.eval_interval) % 10 == 0, {'beta':0, 'alpha':1})
                else:t_hit[k], t_dis[k], t_epi_r[k], t_epi_r_g[k] = t_hit[0], t_dis[0], t_epi_r[0], t_epi_r_g[0] 
                his_r_g[k].append(np.mean(t_epi_r_g[k]))
                for q in range(4):
                    his_hit[k][q].append(np.mean(t_hit[k], 0)[q])
            t_dis = np.mean(t_dis)
            his_dis.append(t_dis)
            his_r.append(np.mean(t_epi_r))
            his_ent.append(np.mean(dist_entropy[-1]))
            his_norm.append(get_v('norm'))
            his_g_norm.append(get_v('grad_norm'))
            his_c_ent.append(get_v('c_ent'))
            his_d_ent.append(get_v('d_ent'))
            draw(his_n_step, [his_dis], 'n of steps', 'dis', "average dis between agents", save_path)
            for k in range(2):
                draw(his_n_step, [his_r, his_r_g[k]], 'n of steps', 'rewards', 'rewards in 1 episode '+ "coop eps = " +str(k), save_path, label=['γ = 1', 'γ = 0.99'])
                draw(his_n_step, his_hit[k], 'n of steps', 'rate', "events' rate (coop eps = "+str(k)+")", save_path, label=['split', 'catch', 'agent', 'cooperate'])
            draw(his_n_step, [his_ent   ], 'n of steps', 'entropy', 'average entropy of the policy in 1 episode during training', save_path)
            draw(his_n_step, [his_norm  ], 'n of steps', 'paramnorm', "parameters' norm (weight averaged)", save_path)
            draw(his_n_step, [his_g_norm], 'n of steps', 'gradnorm', "gradients' norm (weight averaged)", save_path)
            draw(his_n_step, [his_d_ent, his_c_ent], 'n of steps', 'entropy', "average entropy of the policy in 1 episode during training (d & c)", save_path, ['discrete', 'continuous'])
            np.save(os.path.join(save_path, "his_n_step.npy"), np.array(his_n_step))
            np.save(os.path.join(save_path, "his_r.npy"), np.array(his_r))
            np.save(os.path.join(save_path, "his_r_g.npy"), np.array(his_r_g))
            np.save(os.path.join(save_path, "his_hit.npy"), np.array(his_hit))
            np.save(os.path.join(save_path, "his_dis.npy"), np.array(his_dis))
            np.save(os.path.join(save_path, "his_ent.npy"), np.array(his_ent))
            np.save(os.path.join(save_path, "his_norm.npy"), np.array(his_norm))
            np.save(os.path.join(save_path, "his_g_norm.npy"), np.array(his_g_norm))
            np.save(os.path.join(save_path, "his_c_ent.npy"), np.array(his_c_ent))
            np.save(os.path.join(save_path, "his_d_ent.npy"), np.array(his_d_ent))

if __name__ == "__main__":
    main()
