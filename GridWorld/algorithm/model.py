import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.distributions import Bernoulli, Categorical, DiagGaussian
from utils.util import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, num_agents, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
                self.base = base(num_agents, obs_shape, **base_kwargs)
            elif len(obs_shape) == 1:
                base = MLPBase
                self.base = base(num_agents, obs_shape[0], **base_kwargs)
            else:
                raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def is_naive_recurrent(self):
        return self.base.is_naive_recurrent
        
    @property
    def is_lstm(self):
        return self.base.is_lstm

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, share_inputs, inputs, rnn_hxs_actor, rnn_hxs_critic, masks):
        raise NotImplementedError

    def act(self, share_inputs, inputs, rnn_hxs_actor, rnn_hxs_critic, rnn_c_actor, rnn_c_critic, masks, deterministic=False):
        value, actor_features, rnn_hxs_actor, rnn_hxs_critic, rnn_c_actor, rnn_c_critic = self.base(share_inputs, inputs, rnn_hxs_actor, rnn_hxs_critic, rnn_c_actor, rnn_c_critic, masks)
        
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs_actor, rnn_hxs_critic, rnn_c_actor, rnn_c_critic

    def get_value(self, share_inputs, inputs, rnn_hxs_actor, rnn_hxs_critic, rnn_c_actor, rnn_c_critic, masks):
        value, _, _, _,_ ,_ = self.base(share_inputs, inputs, rnn_hxs_actor, rnn_hxs_critic, rnn_c_actor, rnn_c_critic, masks)
        return value

    def evaluate_actions(self, share_inputs, inputs, rnn_hxs_actor, rnn_hxs_critic, rnn_c_actor, rnn_c_critic, masks, action):
        value, actor_features, rnn_hxs_actor, rnn_hxs_critic, rnn_c_actor, rnn_c_critic = self.base(share_inputs, inputs, rnn_hxs_actor, rnn_hxs_critic, rnn_c_actor, rnn_c_critic, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs_actor, rnn_hxs_critic, rnn_c_actor, rnn_c_critic


class NNBase(nn.Module):
    def __init__(self, lstm, naive_recurrent, recurrent, num_agents, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent
        self._naive_recurrent = naive_recurrent
        self._lstm=lstm

        if lstm:
            self.lstm = nn.LSTM(recurrent_input_size, hidden_size)
            self.lstm_critic = nn.LSTM(recurrent_input_size, hidden_size)
            for name, param in self.lstm.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)
            for name, param in self.lstm_critic.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)
        if recurrent or naive_recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            self.gru_critic = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)
            for name, param in self.gru_critic.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def is_naive_recurrent(self):
        return self._naive_recurrent
        
    @property
    def is_lstm(self):
        return self._lstm

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent or self._naive_recurrent or self._lstm:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            #x= self.gru(x.unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)          
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = torch.transpose(x.view(N, T, x.size(1)),0,1)
            
            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)

            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                rnn_scores, hxs = self.gru( x[start_idx:end_idx], hxs * masks[start_idx].view(1, -1, 1))                  
                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            
            x = torch.cat(outputs, dim=0)
            x= torch.transpose(x,0,1)

            # flatten
            x = x.reshape(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs
        
    def _forward_gru_critic(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru_critic(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            #x = self.gru_critic(x.unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = torch.transpose(x.view(N, T, x.size(1)),0,1)

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru_critic(x[start_idx:end_idx], hxs * masks[start_idx].view(1, -1, 1))
                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            x= torch.transpose(x,0,1)
            # flatten
            x = x.reshape(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs
        
    def _forward_lstm(self, x, hxs, c, masks):
        if x.size(0) == hxs.size(0):
            x, (hxs,c) = self.lstm(x.unsqueeze(0), ((hxs * masks).unsqueeze(0), c.unsqueeze(0)))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
            c = c.squeeze(0)           
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = torch.transpose(x.view(N, T, x.size(1)),0,1)

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            c = c.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                #rnn_scores, hxs = self.gru( x[start_idx:end_idx],hxs * masks[start_idx].view(1, -1, 1))                  
                rnn_scores, (hxs, c) = self.lstm(x[start_idx:end_idx], (hxs * masks[start_idx].view(1, -1, 1),c))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            x= torch.transpose(x,0,1)
            # flatten
            x = x.reshape(T * N, -1)
            hxs = hxs.squeeze(0)
            c.squeeze(0)

        return x, hxs, c
        
    def _forward_lstm_critic(self, x, hxs, c, masks):
        if x.size(0) == hxs.size(0):
            x, (hxs, c) = self.lstm_critic(x.unsqueeze(0), ((hxs * masks).unsqueeze(0), c.unsqueeze(0)))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
            c = c.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = torch.transpose(x.view(N, T, x.size(1)),0,1)

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            c = c.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                rnn_scores, (hxs,c) = self.lstm_critic(x[start_idx:end_idx], (hxs * masks[start_idx].view(1, -1, 1),c))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            x= torch.transpose(x,0,1)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)
            c = c.squeeze(0)

        return x, hxs, c


class CNNBase(NNBase):
    def __init__(self, num_agents, inputs, lstm = False, naive_recurrent=False, recurrent=False, hidden_size=64):
        super(CNNBase, self).__init__(lstm, naive_recurrent, recurrent, num_agents, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
                               
        
        num_inputs = inputs[0]
        num_image = inputs[1]

        self.actor = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 3, stride=1)), nn.ReLU(), 
            Flatten(),
            init_(nn.Linear(32 * (num_image-3+1) * (num_image-3+1), hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU())

        self.critic = nn.Sequential(
            init_(nn.Conv2d(num_inputs * num_agents, 32, 3, stride=1)), nn.ReLU(), 
            Flatten(),
            init_(nn.Linear(32 * (num_image-3+1) * (num_image-3+1), hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU(),
            )

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, share_inputs, inputs, rnn_hxs_actor, rnn_hxs_critic, rnn_c_actor, rnn_c_critic, masks):
        x = inputs / 255.0
        share_x = share_inputs / 255.0
        
        hidden_critic = self.critic(share_x)
        hidden_actor = self.actor(x)

        if self.is_recurrent or self.is_naive_recurrent:
            hidden_actor, rnn_hxs_actor = self._forward_gru(hidden_actor, rnn_hxs_actor, masks)
            hidden_critic, rnn_hxs_critic = self._forward_gru_critic(hidden_critic, rnn_hxs_critic, masks)
            
        if self.is_lstm:
            hidden_actor, rnn_hxs_actor, rnn_c_actor = self._forward_lstm(hidden_actor, rnn_hxs_actor, rnn_c_actor, masks)
            hidden_critic, rnn_hxs_critic, rnn_c_critic = self._forward_lstm_critic(hidden_critic, rnn_hxs_critic, rnn_c_critic, masks)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs_actor, rnn_hxs_critic, rnn_c_actor, rnn_c_critic


class MLPBase(NNBase):
    def __init__(self, num_agents, num_inputs, lstm = False, naive_recurrent = False, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(lstm, naive_recurrent, recurrent, num_agents, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs * num_agents, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, share_inputs, inputs, rnn_hxs_actor, rnn_hxs_critic, rnn_c_actor, rnn_c_critic, masks):
        share_x = share_inputs
        x = inputs
        
        hidden_critic = self.critic(share_x)
        hidden_actor = self.actor(x)

        if self.is_recurrent or self.is_naive_recurrent:
            hidden_actor, rnn_hxs_actor = self._forward_gru(hidden_actor, rnn_hxs_actor, masks)
            hidden_critic, rnn_hxs_critic = self._forward_gru_critic(hidden_critic, rnn_hxs_critic, masks)
            
        if self.is_lstm:
            hidden_actor, rnn_hxs_actor, rnn_c_actor = self._forward_lstm(hidden_actor, rnn_hxs_actor, rnn_c_actor, masks)
            hidden_critic, rnn_hxs_critic, rnn_c_critic = self._forward_lstm_critic(hidden_critic, rnn_hxs_critic, rnn_c_critic, masks)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs_actor, rnn_hxs_critic, rnn_c_actor, rnn_c_critic
