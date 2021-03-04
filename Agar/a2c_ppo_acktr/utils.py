import glob
import os

import torch
import torch.nn as nn

from a2c_ppo_acktr.envs import VecNormalize

def get_p_and_g_mean_norm(it):

    size = 0
    su_p = 0
    su_g = 0
    for x in it:
        if x.grad is None:continue
        size += x.numel()
        su_p += x.norm() * x.numel()
        su_g += x.grad.norm() * x.numel()
    return su_p / size, su_g / size

def get_p_and_g_mean_norm2(it):

    size = 1e-8
    su_p = 0
    su_g = 0
    for x in it:
        if x.grad is None:continue
        size += 1.
        su_p += x.norm()
        su_g += x.grad.norm()
    return su_p / size, su_g / size

# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)
