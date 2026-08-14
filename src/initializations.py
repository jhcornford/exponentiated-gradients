import math
import warnings

from torch import Tensor
import torch
from typing import Optional as _Optional


def normal_(
    tensor: Tensor,
    gain: float = 1.0,
    mode: str = "fan_in",
    generator: _Optional[torch.Generator] = None,
    **ignored
):
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = torch.nn.init._calculate_correct_fan(tensor, mode)
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std, generator=generator)

def uniform_(
    tensor: Tensor,
    gain: float = 1.0,
    mode: str = "fan_in",
    generator: _Optional[torch.Generator] = None,
    **ignored
):
    # I'm not sure why this is here in PyTorch
    if torch.overrides.has_torch_function_variadic(tensor):
        return torch.overrides.handle_torch_function(
            uniform_,
            (tensor,),
            tensor=tensor,
            gain=gain,
            mode=mode,
            generator=generator)

    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = torch.nn.init._calculate_correct_fan(tensor, mode)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound, generator=generator)

def calc_ln_mu_sigma(mean, var):
    "Given desired mean and var returns ln mu and sigma"
    mu_ln = math.log(mean ** 2 / math.sqrt(mean ** 2 + var))
    sigma_ln = math.sqrt(math.log(1 + (var / mean ** 2)))
    return mu_ln, sigma_ln

def lognormal_(
    tensor: Tensor,
    gain: float = 1.0,
    mode: str = "fan_in",
    generator: _Optional[torch.Generator] = None,
    mean_std_ratio: float = 1,
    **ignored
):
    """
    Initializes the tensor with a log normal distribution * {1,-1}. 

    Arguments:
        tensor: torch.Tensor, the tensor to initialize
        gain: float, the gain to use for the initialization stddev calulation.
        mode: str, the mode to use for the initialization. Options are 'fan_in', 'fan_out'
        generator: optional torch.Generator, the random number generator to use. 
        mean_std_ratio: float, the ratio of the mean to std for log_normal initialization.

    Note this function draws from a log normal distribution with mean = mean_std_ratio * std
    and then multiplies the tensor by a random Rademacher dist. variable (impl. with bernoulli). 
    This induces the need to correct the ln std dev, as the final symmetrical distribution
    will have variance = mu^2 + sigma^2 = (1 + mean_std_ratio^2) * sigma^2. Where sigma, mu are
    the log normal distribution parameters.
    """
    fan = torch.nn.init._calculate_correct_fan(tensor, mode)
    std = gain / math.sqrt(fan)
    std /= (1+mean_std_ratio**2)**0.5 # Adjust for multiplication with bernoulli  
    mu, sigma = calc_ln_mu_sigma(std * mean_std_ratio, std ** 2)
    with torch.no_grad():
        tensor.log_normal_(mu, sigma, generator=generator)
        return tensor.mul_(2 * torch.bernoulli(0.5 * torch.ones_like(tensor), generator=generator) - 1)


def re_init_network(net, all_modules=False, init='normal', gain=1/math.sqrt(3), 
                    mean_std_ratio:_Optional[float]=1.0, 
                    generator:_Optional[torch.Generator]=None):
    """
    Arguments:
        net: torch.nn.Module, the network to re-init
        all_modules: bool, if true also re-inits the linear output weights.
        init: str, the initialization to use. Options are 'normal', 'log_normal', 'uniform'
        gain: float, the gain to use for the initialization stddev calulation.
        mean_std_ratio: float, only used log_normal initialization: the ratio of the mean to std 
        generator: optional torch.Generator, the random number generator to use. 

    Note this function does not change biases.

    Pytorch default initialization bounds are based on a "good bug", where the uniform bounds are set
    to be the value of the std dev of weights for a given fan in/out. Therefore to replicate pytorch
    default init use gain = 1/root(3).  
    """
    if init == 'normal':
        init_func = normal_
    elif init == 'log_normal':
        init_func = lognormal_
    elif init == 'uniform':
        init_func = uniform_
    else:
        raise ValueError(f'{init} is not implemented')

    for m in net.modules():
        if isinstance(m, torch.nn.RNNCellBase):
            # fan out as pytorch default init bounds are based on hidden size
            init_func(m.weight_hh, gain=gain, mode="fan_out", mean_std_ratio=mean_std_ratio, generator=generator)
            init_func(m.weight_ih, gain=gain, mode="fan_out", mean_std_ratio=mean_std_ratio, generator=generator)
        elif isinstance(m, torch.nn.Linear) and all_modules:
            init_func(m.weight, gain=gain, mode="fan_in", mean_std_ratio=mean_std_ratio, generator=generator)
