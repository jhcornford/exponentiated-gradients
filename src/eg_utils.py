import torch
import torch.nn as nn
import math
import random
import numpy as np

from typing import List, Optional
from torch import Tensor
from torch.nn import RNNCellBase
from torch.optim.optimizer import Optimizer, required

def set_seed_all(seed:int):
    """
    Sets all random states
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

class SplitBias(nn.Module):
    def __init__(self, base_layer):
        super().__init__()
        self.base_layer = base_layer
        self.has_bias = base_layer.bias is not None
        if self.has_bias:
            self.bias_pos = nn.Parameter(torch.empty_like(self.base_layer.bias))
            self.bias_neg = nn.Parameter(torch.empty_like(self.base_layer.bias))
            del self.base_layer.bias
            self.base_layer.bias = 0

        self.reset_parameters()    

    def reset_parameters(self) -> None:
        if self.has_bias:
            if len(self.base_layer.weight.shape) == 1:
                # norm layers, biases usually initialized with zeros
                bound = 1e-4 
            else:
                # conv & linear layers, biases usually initialized with uniform
                # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
                # same variance as for non-split bias
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.base_layer.weight)
                if fan_in > 0: bound = 1 / math.sqrt(fan_in) 
                else: raise
            nn.init.uniform_(self.bias_pos, 0, math.sqrt(2) * bound)
            nn.init.uniform_(self.bias_neg, -bound * math.sqrt(2), 0)

    def forward(self, x):
        self.base_layer.bias = self.bias_pos + self.bias_neg
        return self.base_layer(x)

class SplitBiasRNNCell(nn.Module):
    def __init__(self, base_layer):
        super().__init__()
        self.base_layer = base_layer
        assert isinstance(self.base_layer, RNNCellBase)
        self.has_bias = self.base_layer.bias
        if self.has_bias:
            self.bias_pos_ih = nn.Parameter(torch.empty_like(self.base_layer.bias_ih))
            self.bias_neg_ih = nn.Parameter(torch.empty_like(self.base_layer.bias_ih))
            self.bias_pos_hh = nn.Parameter(torch.empty_like(self.base_layer.bias_hh))
            self.bias_neg_hh = nn.Parameter(torch.empty_like(self.base_layer.bias_hh))
            del self.base_layer.bias_ih
            del self.base_layer.bias_hh
            self.base_layer.bias_ih = 0
            self.base_layer.bias_hh = 0

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.has_bias:
            # see RNNCellBase https://pytorch.org/docs/stable/_modules/torch/nn/modules/rnn.html
            # for weight in self.parameters():
            #     init.uniform_(weight, -stdv, stdv)
            stdv = 1.0 / math.sqrt(self.base_layer.hidden_size) if self.base_layer.hidden_size > 0 else 0
            nn.init.uniform_(self.bias_pos_ih, 0, math.sqrt(2) * stdv)
            nn.init.uniform_(self.bias_neg_ih, -stdv * math.sqrt(2), 0)
            nn.init.uniform_(self.bias_pos_hh, 0, math.sqrt(2) * stdv)
            nn.init.uniform_(self.bias_neg_hh, -stdv * math.sqrt(2), 0)

    def forward(self, x, hx):
        self.base_layer.bias_ih = self.bias_pos_ih + self.bias_neg_ih
        self.base_layer.bias_hh = self.bias_pos_hh + self.bias_neg_hh
        return self.base_layer(x, hx)

    
def set_split_bias(module):
    attr_to_change = dict()
    for name, child in module.named_children():
        if len(list(child.children())) > 0:
            set_split_bias(child)
        else:
            if hasattr(child, 'bias') and child.bias is not None:
                if isinstance(child, nn.RNNCellBase):
                    attr_to_change[name] = SplitBiasRNNCell(child)
                else:
                    attr_to_change[name] = SplitBias(child)
    for name, value in attr_to_change.items():
        setattr(module, name, value)


class SGD(Optimizer):
    """
    This is a slightly stripped down & modified version of torch.optim.SGD
     
    See https://github.com/pytorch/pytorch/blob/main/torch/optim/sgd.py 

    update_alg can choose between gradient descent ('gd'), exponentiated gradient ('eg'),
    and 'eg_unsigned', an ablation of eg that does not multiply the update by the weight sign.
    For 'gd', freeze_gd_signs=True implements sign-constrained gradient descent.
    To avoid storing signs for each parameter, updates that cross 0 fix the weight
    at +-freeze_gd_signs_th instead (so the next param.sign() returns the same sign).
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, update_alg="gd",
                 freeze_gd_signs=False, freeze_gd_signs_th=1e-18):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if update_alg not in ["gd", "eg", "eg_unsigned"]:
            raise ValueError("Invalid update_alg value: {}".format(update_alg))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        update_alg=update_alg, 
                        freeze_gd_signs=freeze_gd_signs,
                        freeze_gd_signs_th=freeze_gd_signs_th)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []

            # this is from optim.sgd._init_group
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            sgd(params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=group['weight_decay'],
                momentum= group['momentum'],
                lr=group['lr'],
                dampening=group['dampening'],
                nesterov=group['nesterov'],
                update_alg=group['update_alg'],
                freeze_gd_signs=group['freeze_gd_signs'],
                freeze_gd_signs_th=group['freeze_gd_signs_th'])

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss


def sgd(params: List[Tensor],
        d_p_list: List[Tensor], 
        momentum_buffer_list: List[Optional[Tensor]],
        *,
        weight_decay: float, 
        momentum: float, 
        lr: float, 
        dampening: float, 
        nesterov: bool, 
        update_alg: str,
        freeze_gd_signs: bool,
        freeze_gd_signs_th: float):

    for i, param in enumerate(params):

        d_p = d_p_list[i]
        if weight_decay != 0:
            if update_alg =="gd":
                d_p = d_p.add(param, alpha=weight_decay)
            elif update_alg == "eg":
                d_p = d_p.add(param.sign(), alpha=weight_decay)
            elif update_alg == "eg_unsigned":
                d_p = d_p.add(param.sign().abs(), alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        if update_alg == "gd" and freeze_gd_signs:
            s = param.sign()
            param.add_(d_p, alpha=-lr)
            # if s and param signs are different, s * param becomes negative
            flip_idx = (s * param) < 0
            param[flip_idx] = freeze_gd_signs_th * s[flip_idx] # setting to a non-zero value so the weight keeps the sign
        elif update_alg == 'gd':
            param.add_(d_p, alpha=-lr)
        elif update_alg == "eg":
            # multiply by sign to ensure that the update is in the correct direction
            # if weight is neg, and grad is neg (so pos update) instead the weight will become more negative
            param.mul_(torch.exp(param.sign() * d_p * -lr))
        elif update_alg == "eg_unsigned":
            # ablation: eg but not multiply by sign 
            param.mul_(torch.exp(d_p * -lr))

def max_singular_value(W, num_iters=50):
    v = torch.randn(W.size(1), device=W.device)
    v = v / torch.norm(v)
    
    WtW = W.T @ W
    for _ in range(num_iters):
        v = torch.mv(WtW, v)
        v = v / torch.norm(v)
    
    sigma_max = torch.norm(torch.mv(W, v))
    return sigma_max
