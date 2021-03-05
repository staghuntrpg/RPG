import numpy as np
import torch
import time

from agar.Env import AgarEnv
from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs

render = True

def evaluate(args, agents, ob_rms, env_name, seed, num_processes, eval_log_dir,
             device, n_agent, out_file):
    
    e_env = AgarEnv(args, eval = True)
    eval_episode_rewards = []

    obs = e_env.reset()
    for i in range(n_agent):
        obs['t'+str(i)] = torch.Tensor(obs['t'+str(i)]).to(device)
    eval_recurrent_hidden_states = [torch.zeros(1,
        agents[0].actor_critic.recurrent_hidden_state_size, device=device) for i in range(n_agent)]
    eval_masks = [torch.zeros(1, 1, device=device) for i in range(n_agent)]
    action = [[] for i in range(n_agent)]
    step = 0
    while len(eval_episode_rewards) < 10:
        step += 1
        if render:
            e_env.render(0, mode = 'rgb_array', name = str(len(eval_episode_rewards)))
        
        for i in range(n_agent):
            with torch.no_grad():
                _, action[i], r, eval_recurrent_hidden_states[i] = agents[0].actor_critic.act(
                    obs['t'+str(i)].reshape(1, -1),
                    eval_recurrent_hidden_states[i],
                    eval_masks[i],
                    deterministic=True)

        # Obser reward and next obs
        obs, r, done, infos = e_env.step(torch.cat(action, -1).reshape(-1).cpu())
        
        if len(eval_episode_rewards) == 0:
            print(obs['t0'][-5:])
            out_file.write(str(obs['t0'][-5:])+'\n')
            print('action & reward in evaluation', action, r)
            out_file.write('action & reward in evaluation '+str(action)+' '+str(r)+'\n')
        for i in range(n_agent):
            obs['t'+str(i)] = torch.Tensor(obs['t'+str(i)]).to(device)

        for i in range(n_agent):
            eval_masks[i] = torch.tensor(
                [0.0] if done[i] else [1.0],
                dtype=torch.float32,
                device=device)
        
        for i in range(n_agent):
            if 'episode' in infos[i].keys():
                eval_episode_rewards.append(infos[i]['episode']['r'])
        done = (np.array(done) != 0).all()
        if done:
            step = 0
            e_env.close()
            e_env = AgarEnv(args, eval = True)
            obs = e_env.reset()
            for i in range(n_agent):
                obs['t'+str(i)] = torch.Tensor(obs['t'+str(i)]).to(device)
    ss = (" Evaluation using {} episodes: mean reward {:.5f}\n".format(len(eval_episode_rewards), np.mean(eval_episode_rewards)))
    print(ss)
    out_file.write(ss+'\n')
    print('var: ',np.var(eval_episode_rewards) / 10)
    out_file.write('var: '+str(np.var(eval_episode_rewards) / 10)+'\n')
    print(eval_episode_rewards)
    out_file.write(str(eval_episode_rewards)+'\n')
    return np.mean(eval_episode_rewards)
