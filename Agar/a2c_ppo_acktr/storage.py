import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from gv import *
import numpy as np

def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])

class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_size, action_size,
                 recurrent_hidden_state_size):
        self.obs = torch.zeros(num_steps + 1, num_processes, obs_size)
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.actions = torch.zeros(num_steps, num_processes, action_size)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # bad_masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)
        self.high_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        self.high_masks = self.high_masks.to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks, high_masks = None): 
               # after init, in main the step 0 is fed with first obs, then the first insert will be put in step 1
        #mask[i] means there is no transition between step i - 1 and step i (done -> 0)
        #bad_mask: exceed_time_limit -> 0
        #when bad_mask = 0, mask must = 0
        #hidden[step] and obs[step] are faced at the same time.
        #action[step], rewards[step] happens between obs[step] and obs[step + 1]
        #high_mask[i] bans actions on obs[i]'s training
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        if high_masks is not None:self.high_masks[self.step + 1].copy_(high_masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])
        self.high_masks[0].copy_(self.high_masks[-1])

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True):
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae # masks = not done
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                        + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
                # when bad mask = 0 (also mask = 0) returns[step] = value_pred, no value train
        else:
            print("Not implement 'use_proper_time_limits = False'")
            exit(0)

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        perm = torch.randperm(num_mini_batch * num_processes)
        length = self.num_steps // num_mini_batch
        assert self.num_steps % num_mini_batch == 0, (
            "num_mini_batch should devide num_steps (in storage)")
        for i in range(num_mini_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            high_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for j in range(num_processes):
                ind = perm[i * num_processes + j] // num_mini_batch
                pos = perm[i * num_processes + j] % num_mini_batch
                pos = pos * length
                obs_batch.append(self.obs[pos:pos + length, ind])
                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[pos:pos+1, ind])
                actions_batch.append(self.actions[pos:pos + length, ind])
                value_preds_batch.append(self.value_preds[pos:pos + length, ind])
                return_batch.append(self.returns[pos:pos + length, ind])
                masks_batch.append(self.masks[pos:pos + length, ind])
                high_masks_batch.append(self.high_masks[pos:pos + length, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[pos:pos + length, ind])
                adv_targ.append(advantages[pos:pos + length, ind])

            T, N = length, num_processes
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            high_masks_batch = torch.stack(high_masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            high_masks_batch = _flatten_helper(T, N, high_masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)
            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, high_masks_batch, old_action_log_probs_batch, adv_targ
