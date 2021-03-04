import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gv import *
from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init
from copy import deepcopy
device = get_v('device')

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        base = MLPBase
        obs_size = get_v('obs_size')
        self.base = base(obs_size, **base_kwargs)
        
        self.dist = nn.ModuleList([DiagGaussian(self.base.output_size, 2), Categorical(self.base.output_size, 2)])

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist, action, action_log_probs = [None, None], [None, None], [None, None]
        for i in range(2):
            dist[i] = self.dist[i](actor_features)

            if deterministic:
                action[i] = dist[i].mode().float()
            else:
                action[i] = dist[i].sample().float()

            action_log_probs[i] = dist[i].log_probs(action[i])
        return value, torch.cat(action, -1), torch.sum(torch.cat(action_log_probs, -1), -1, keepdim = True), rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, high_masks = None):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        a, b = action.split((2, 1), -1)
        b = b.long()
        action = [a, b]
        dist, action_log_probs, dist_entropy = [None, None], [None, None], [None, None]
        for i in range(2):
            dist[i] = self.dist[i](actor_features)
            action_log_probs[i] = dist[i].log_probs(action[i])
            if high_masks is not None:
                dist_entropy[i] = (dist[i].entropy().reshape(-1) * high_masks.reshape(-1)).sum() / high_masks.sum()
            else:dist_entropy[i] = dist[i].entropy().mean()
        set_v('d_ent', dist_entropy[1].detach().cpu().numpy() * 2.)
        set_v('c_ent', dist_entropy[0].detach().cpu().numpy() * 0.5)
        return value, torch.sum(torch.cat(action_log_probs, -1), -1, keepdim = True), dist_entropy[0] * 0.5 + dist_entropy[1] * 2, rnn_hxs
        
def std_init(t):

    for name, param in t.named_parameters():
        if 'bias' in name:
                nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param)

class MHAT2(nn.Module): # multi-head attention

    def __init__(self, q_size, k_size, hidden_size, out_size, n_head = 4):
        super(MHAT2, self).__init__()
        self.n_head = n_head
        self.hidden_size = hidden_size
        self.k_size = k_size
        self.fc_q = nn.ModuleList([nn.Linear(q_size, hidden_size, bias = True) for i in range(n_head)])
        self.fc_v = nn.ModuleList([nn.Linear(k_size, hidden_size, bias = False) for i in range(n_head)])
        self.fc_k = nn.ModuleList([nn.Linear(k_size, hidden_size, bias = False) for i in range(n_head)])
        self.fc_o = nn.Linear(n_head * hidden_size, out_size, bias = False)
        self.ln = nn.LayerNorm(normalized_shape = (n_head * hidden_size, ))
        for i in range(self.n_head):
            std_init(self.fc_k[i])
            std_init(self.fc_q[i])
            std_init(self.fc_v[i])
        std_init(self.fc_o)
    
    def forward(self, me, other, mask = None): # size of me and other are [T, N, q_size], [T, N, k, v_size] mask: [T, N, k]
        l_out = [] 
        for i in range(self.n_head):
            k = self.fc_k[i](other) # [T, N, k, h_size]
            v = self.fc_v[i](other)
            q = self.fc_q[i](me)
            a = torch.matmul(q.unsqueeze(-2), k.transpose(-1, -2)) / self.hidden_size ** 0.5 # [T, N, 1, h] [T, N, h, k] [T, N, 1, k]
            a = a.squeeze()
            mask = None
            if mask is not None:
                a = torch.where(mask > 0., a, torch.full_like(a, -1e10))
            a = torch.softmax(a, dim = -1) #[T, N, k]
            out = torch.sum(a.unsqueeze(-1) * v, -2) #[T, N, h]
            l_out.append(out)
        if self.n_head == 1:c = l_out
        else:c = torch.cat(l_out, -1)
        return self.fc_o(torch.relu(self.ln(c)))

class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size, attention = 0):
        super(NNBase, self).__init__()
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self._hidden_size = hidden_size
        self._recurrent = recurrent
        self._attention = attention
        
        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

        if attention:
            self.mhat_b = MHAT2(28, 15, hidden_size // 2, hidden_size // 2, attention) # q k h out head
            self.mhat_f = MHAT2(28, 7, hidden_size // 2, hidden_size // 2, attention)
            self.mhat_v = MHAT2(28, 5, hidden_size // 2, hidden_size // 2, attention)
            self.mhat_o = MHAT2(28, 15, hidden_size // 2, hidden_size // 2, attention) # q k h out head
            self.mhat_p = MHAT2(28, 15, hidden_size // 2, hidden_size // 2, attention) # q k h out head
            self.em = nn.Sequential(init_(nn.Linear(28 + hidden_size // 2 * 5, recurrent_input_size)), nn.ReLU())
        
    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def is_attention(self):
        return self._attention
    
    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        
        n = [10, 5, 5, 10, 10, 10, 5, 5, 10, 10]
        n2 = [15, 7, 5, 15, 15, 1, 1, 1, 1, 1]
        n3 = [n[i] * n2[i] for i in range(10)]
        for i in range(1, 10):
            n3[i] += n3[i-1]
        for i in range(1, len(n)):
            n2[i] += n2[i-1]
        if x.size(0) == hxs.size(0):
            
            N = x.shape[0]
            if self._attention:

                x_ball = x[:,:n3[0]]
                x_food = x[:,n3[0]:n3[1]]
                x_virus = x[:,n3[1]:n3[2]]
                x_oppo = x[:,n3[2]:n3[3]]
                x_ptnr = x[:,n3[3]:n3[4]]
                mask_ball = x[:,n3[4]:n3[5]]
                mask_food = x[:,n3[5]:n3[6]]
                mask_virus = x[:,n3[6]:n3[7]]
                mask_oppo = x[:,n3[7]:n3[8]]
                mask_ptnr = x[:,n3[8]:n3[9]]
                x_glo = x[:,n3[9]:]
                em_ball = self.mhat_b(x_glo, x_ball.view(N, n[0], -1), mask_ball)# [n_pro, n_entity, entity_d]
                em_food = self.mhat_f(x_glo, x_food.view(N, n[1], -1), mask_food)
                em_virus = self.mhat_v(x_glo, x_virus.view(N, n[2], -1), mask_virus)
                em_oppo = self.mhat_o(x_glo, x_oppo.view(N, n[3], -1), mask_oppo)
                em_ptnr = self.mhat_p(x_glo, x_ptnr.view(N, n[4], -1), mask_ptnr)
                x = self.em(torch.cat([em_ball, em_oppo, em_ptnr, em_food, em_virus, x_glo], -1))
            
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            # hxs is (T, N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N) # masks is (num_env_step, n_pro/n_mini)
            
            if self._attention: 


                x_ball = x[:,:,:n3[0]]
                x_food = x[:,:,n3[0]:n3[1]]
                x_virus = x[:,:,n3[1]:n3[2]]
                x_oppo = x[:,:,n3[2]:n3[3]]
                x_ptnr = x[:,:,n3[3]:n3[4]]
                mask_ball = x[:,:,n3[4]:n3[5]]
                mask_food = x[:,:,n3[5]:n3[6]]
                mask_virus = x[:,:,n3[6]:n3[7]]
                mask_oppo = x[:,:,n3[7]:n3[8]]
                mask_ptnr = x[:,:,n3[8]:n3[9]]
                x_glo = x[:,:,n3[9]:]
                em_ball = self.mhat_b(x_glo, x_ball.view(T, N, n[0], -1), mask_ball)
                em_food = self.mhat_f(x_glo, x_food.view(T, N, n[1], -1), mask_food)
                em_virus = self.mhat_v(x_glo, x_virus.view(T, N, n[2], -1), mask_virus)
                em_oppo = self.mhat_o(x_glo, x_oppo.view(T, N, n[3], -1), mask_oppo)
                em_ptnr = self.mhat_p(x_glo, x_ptnr.view(T, N, n[4], -1), mask_ptnr)
                x = self.em(torch.cat([em_ball, em_oppo, em_ptnr, em_food, em_virus, x_glo], -1))

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(T):
                
                temp = x[i].view(1, N, -1)
                rnn_scores, hxs = self.gru(temp, hxs * masks[i].view(1, -1, 1)) 
                outputs.append(rnn_scores)

            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)
                               
        return x, hxs

class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=128, attention = 0):
        super(MLPBase, self).__init__(recurrent, hidden_size, hidden_size, attention)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        else:
            print('only implement recurrent')
            exit(1)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
