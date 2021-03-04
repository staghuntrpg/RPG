import copy
import glob
import os
import time
from draw import draw
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
    s = s.read().replace('MiB','').replace('memory.free','').split('\n')
    for i in range(1, len(s) - 1):
        a.append(int(s[i]))
    print(a)
    best = int(np.argmax(a))
    print('the best GPU is ',best,' with free memories of ',s[best + 1])
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
set_v('user_name', user_name)

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.cuda.set_device(gpu_id)

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation2 import evaluate

def main():
    args = get_args()
    
    load = False
    if args.load_time_label is not None:
        load_time_label = args.load_time_label
        load_path = os.path.join('trained_models', args.load_time_label)
        args=np.load(os.path.join(load_path, 'args.npy'), allow_pickle=True)[0]
        args.total_step = 1e8
        load = True
    else:
        print('error: no load_time_label')
        exit(1)
    
    save_path = os.path.join(args.save_dir, load_time_label)
    save_path = os.path.join(save_path, 'gif')
    try:
        os.makedirs(save_path)
    except OSError:
        pass
    out_file = open(os.path.join(save_path, "gif_out.txt"),"w")
    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(8)
    device = torch.device("cuda" if args.cuda else "cpu")
    set_v('device', device)
    
    n_agent = args.num_controlled_agent

    agent = torch.load(os.path.join(load_path, 'agent.pt'), map_location = 'cuda:'+str(gpu_id))

    
    evaluate(args, agent, None, None, args.seed, args.num_processes, None, device, n_agent, out_file)
    
if __name__ == "__main__":
    main()
