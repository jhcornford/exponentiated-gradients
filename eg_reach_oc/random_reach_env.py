from __future__ import annotations

import torch
import numpy as np
from typing import Any
from motornet.environment import Environment

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')


def tensor(x): return x if torch.is_tensor(x) else torch.tensor(x).to(device).float()
def detach(x): return x.clone().detach().cpu().numpy()

class CustomRandomTargetReach(Environment):
  """A reach to a random target from a random starting position.

  Args:
    network: :class:`motornet.nets.layers.Network` object class or subclass. This is the network that will perform
      the task.
    name: `String`, the name of the task object instance.
    deriv_weight: `Float`, the weight of the muscle activation's derivative contribution to the default muscle L2
      loss.
    **kwargs: This is passed as-is to the parent :class:`Task` class.
  """

  def __init__(
      self,
      effector,
      *args,
      dist_penalty: float = 0.,
      visual_feedback: bool = True,
      return_positive_xy_obs: bool = True,
      return_positive_vel_obs: bool = True,
      positive_noise: bool = True,
      irrel_noise_dims: int = 0,
      irrel_noise_type: str = None,
      irrel_noise_theta: float = 0.01,  # 100ms
      device: torch.device = device,
      **kwargs,
      ):
    self.goal_size = 0.01

    self.visual_feedback = visual_feedback
    self.return_positive_xy_obs = return_positive_xy_obs
    self.return_positive_vel_obs = return_positive_vel_obs
    self.positive_noise = positive_noise
    self.irrel_noise_dims = irrel_noise_dims
    self.irrel_noise_type = irrel_noise_type
    self.irrel_noise_theta = irrel_noise_theta

    # caculate most negative fingertip position for when return_positive_xy_obs is True
    min_x = torch.cos(torch.tensor(np.pi)) * effector.skeleton.l2 + torch.cos(effector.skeleton.pos_upper_bound[0,0]) * effector.skeleton.l1
    min_y = torch.sin(3*torch.tensor(np.pi)/2) * effector.skeleton.l2 + torch.sin(effector.skeleton.pos_upper_bound[0,0]) * effector.skeleton.l1

    self.min_xy_fingertip = torch.nn.Parameter(torch.tensor([min_x, min_y]).reshape(1, 2), requires_grad=False).to(device)
    if self.irrel_noise_dims > 0:
      self.irrel_inputs_buffer = [None]
      if self.irrel_noise_type == 'OUmomentum':
        self.irrel_inputs_speed_buffer = [None]
        self.irrel_inputs_momentum_buffer = [None]
        self.irrel_inputs_buffer_init = [None]

    super().__init__(effector, *args, **kwargs) # calls reset so assign attrs first 
    self.to(device)

    self.obs_noise[:self.skeleton.space_dim] = [0.] * self.skeleton.space_dim  # target info is noiseless
    self.dist_penalty = dist_penalty

  def step(
      self,
      action: torch.Tensor | np.ndarray, # motor command to effector in env, from policy network 
      deterministic: bool = False,
      **kwargs,
    ) -> tuple[torch.Tensor | np.ndarray, bool, bool, dict[str, Any]]:
    """
    Perform one simulation step. This method is likely to be overwritten by any subclass to implement user-defined 
    computations, such as reward value calculation for reinforcement learning, custom truncation or termination
    conditions, or time-varying goals.
    
    Args:
      action: `Tensor` or `numpy.ndarray`, the input drive to the actuators.
      deterministic: `Boolean`, whether observation, action, proprioception, and vision noise are applied.
      **kwargs: This is passed as-is to the :meth:`motornet.effector.Effector.step()` call. This is maily useful to pass
      `endpoint_load` or `joint_load` kwargs.
  
    Returns:
      - The observation vector as `tensor` or `numpy.ndarray`, if the :class:`Environment` is set as differentiable or 
        not, respectively. It has dimensionality `(batch_size, n_features)`.
      - A `numpy.ndarray` with the reward information for the step, with dimensionality `(batch_size, 1)`. This is 
        `None` if the :class:`Environment` is set as differentiable. By default this always returns `0.` in the 
        :class:`Environment`.
      - A `boolean` indicating if the simulation has been terminated or truncated. If the :class:`Environment` is set as
        differentiable, this returns `True` when the simulation time reaches `max_ep_duration` provided at 
        initialization.
      - A `boolean` indicating if the simulation has been truncated early or not. This always returns `False` if the
        :class:`Environment` is set as differentiable.
      - A `dictionary` containing this step's information.
    """
    
    self.elapsed += self.dt

    action = action if torch.is_tensor(action) else torch.tensor(action, dtype=torch.float32).to(self.device)
    noisy_action = action
    if deterministic is False:
      noisy_action = self.apply_noise(noisy_action, noise=self.action_noise)
    
    self.effector.step(noisy_action, **kwargs)

    err = self.states["fingertip"] - self.goal # x, y error
    #dist = np.sqrt(np.square(self.detach(err)).sum()) # euclidean distance error, query not detach
    dist = torch.sqrt(torch.square(err).sum())
    if dist > self.goal_size: # l1 distance of error?
      reward_dist = -torch.abs(err).mean(dim=-1)
    else:
      reward_dist = 0.
      #self.time_in_target += self.dt # dt should be a vector over the batch (masked by in target)
    #reward_ctrl = - torch.square(noisy_action).mean(dim=-1) # this is the muscle 
    reward = self.dist_penalty * reward_dist #+ self.ctrl_penalty * reward_ctrl # reward_dist + reward_ctrl

    self.goal = self.goal.clone() # why clone?
    info = { 
      "states": self._maybe_detach_states(),
      "actions": action,
      "noisy actions": noisy_action,
      "reward_dist": reward_dist,
      #"reward_ctrl": reward_ctrl,
      "goal": self.goal if self.differentiable else self.detach(self.goal),
      }

    # step the irrelevant noise inputs too
    if self.irrel_noise_dims > 0:
      # OU-style
      if self.irrel_noise_type == 'OU':
        self.irrel_inputs_buffer[0] = (1.0 - self.irrel_noise_theta) * self.irrel_inputs_buffer[0] + \
                                      np.sqrt(2 * self.irrel_noise_theta) * torch.randn(action.shape[0],
                                                                                        self.irrel_noise_dims)
        self.irrel_inputs_buffer[0] = torch.relu(self.irrel_inputs_buffer[0])
      elif self.irrel_noise_type == 'OUmomentum':  # now with a bounce back from 0
        self.irrel_inputs_momentum_buffer[0][self.irrel_inputs_buffer[0] == 0] = 0.0
        self.irrel_inputs_speed_buffer[0][self.irrel_inputs_buffer[0] == 0] = 0.0

        self.irrel_inputs_momentum_buffer[0] = (1.0 - self.irrel_noise_theta) * self.irrel_inputs_momentum_buffer[0] + \
                                      np.sqrt(2 * self.irrel_noise_theta) * torch.randn(action.shape[0],
                                                                                        self.irrel_noise_dims)
        self.irrel_inputs_speed_buffer[0] += 0.1 * self.irrel_inputs_momentum_buffer[0] + \
                                      0.1 * (self.irrel_inputs_buffer_init[0] - self.irrel_inputs_buffer[0]) + \
                                             10 * (0.01 - self.irrel_inputs_buffer[0]) * (self.irrel_inputs_buffer[0] <= 0.01)
        self.irrel_inputs_buffer[0] += 0.1 * self.irrel_inputs_speed_buffer[0]
        self.irrel_inputs_buffer[0] = torch.relu(self.irrel_inputs_buffer[0])
      else:
        # fallback: unbounded walk, no relu clamp so can go negative
        self.irrel_inputs_buffer[0] += 0.05 * torch.randn(action.shape[0], self.irrel_noise_dims)


    obs = self.get_obs(action=noisy_action) # get the new set of observations
    truncated  = False # if self.differentiable else bool(self.time_in_target >= 0.5) 
    # self.max_ep_duration is the time limit for the episode
    terminated = (self.elapsed >= self.max_ep_duration) or truncated
    return obs, reward, terminated, truncated, info
  
  def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
    """
    Uses the :meth:`Environment.reset()` method of the parent class :class:`Environment` that can be overwritten to 
    change the returned data. Here the goals (`i.e.`, the targets) are drawn from a random uniform distribution across
    the full joint space.

    note options dict is gym interface to pass options to the reset method
    """
    self._set_generator(seed=seed)

    options = {} if options is None else options
    batch_size: int = options.get('batch_size', 1)
    eval_mode: bool = options.get('eval_mode', False)
    joint_state: torch.Tensor | np.ndarray | None = options.get('joint_state', None)
    deterministic: bool = options.get('deterministic', False)
    
    if joint_state is not None:
      joint_state_shape = np.shape(self.detach(joint_state))
      if joint_state_shape[0] > 1:
        batch_size = joint_state_shape[0]
    else:
      joint_state = self.q_init # default joint state is the initial joint state

    if eval_mode:
      print("Eval circle targets")
      batch_size = 8
      thetas = np.array(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
      r = 0.2 # 20 cm radius
      joint_state = tensor(np.deg2rad([45., 90., 0., 0.])).reshape(1, -1) # initial joint state, shoulder, elbow, s vel, e vel
      p_init = self.joint2cartesian(tensor(np.deg2rad([45., 90., 0., 0.]))).reshape(1, -1)
      x = r * tensor(np.cos(thetas))[:, None] + p_init[0, 0]
      y = r * tensor(np.sin(thetas))[:, None] + p_init[0, 1]*0.9 # add to initial position to centre targets around init pos
      self.goal = torch.concat([x, y], dim=-1)
    else:
      joint_state = self.q_init
      self.goal = self.joint2cartesian(self.effector.draw_random_uniform_states(batch_size)).chunk(2, dim=-1)[0] # first 2 bec position
    self.goal = self.goal.to(self.device)

    self.effector.reset(options={"batch_size": batch_size, "joint_state": joint_state}) # if joint state is None effector starts from random state
    self.elapsed = 0.

    action = torch.zeros((batch_size, self.action_space.shape[0])).to(self.device)
    self.obs_buffer["proprioception"] = [self.get_proprioception()] * len(self.obs_buffer["proprioception"])
    self.obs_buffer["vision"] = [self.get_vision().to(self.device)] * len(self.obs_buffer["vision"])
    self.obs_buffer["action"] = [action] * self.action_frame_stacking
    self.send_obs_buffer_to_device()
    self.irrel_inputs_buffer = [torch.rand(batch_size, self.irrel_noise_dims)] if self.irrel_noise_dims > 0 else []
    self.irrel_inputs_buffer_init = [self.irrel_inputs_buffer[0].clone()] if self.irrel_noise_dims > 0 else []
    self.irrel_inputs_speed_buffer = [0.01 * (torch.rand(batch_size, self.irrel_noise_dims) - 0.5)] if self.irrel_noise_dims > 0 else []
    self.irrel_inputs_momentum_buffer = [torch.zeros_like(self.irrel_inputs_buffer[0])] if self.irrel_noise_dims > 0 else []
    action = action if self.differentiable else self.detach(action)
    
    obs = self.get_obs(deterministic=deterministic)
    info = {
      "states": self._maybe_detach_states(), # states are the joint, cartesian, muscle lengths? 
      "actions": action,
      "noisy actions": action,
      "reward_dist": None,
      #"reward_ctrl": None,
      "goal": self.goal if self.differentiable else self.detach(self.goal),
      }
    return obs, info
  
  @torch.no_grad()
  def grad_reset(self) -> tuple[Any, dict[str, Any]]:
    self.goal = self.goal.clone()

    self.effector.grad_reset() # if joint state is None effector starts from random state

    self.obs_buffer["proprioception"] = detach_tensors_in_structure(self.obs_buffer["proprioception"])
    self.obs_buffer["vision"] = detach_tensors_in_structure(self.obs_buffer["vision"])
    self.obs_buffer["action"] = detach_tensors_in_structure(self.obs_buffer["action"])
    self.irrel_inputs_buffer = detach_tensors_in_structure(self.irrel_inputs_buffer)
    self.irrel_inputs_buffer_init = detach_tensors_in_structure(self.irrel_inputs_buffer_init)
    self.irrel_inputs_speed_buffer = detach_tensors_in_structure(self.irrel_inputs_speed_buffer)
    self.irrel_inputs_momentum_buffer = detach_tensors_in_structure(self.irrel_inputs_momentum_buffer)

  
  def send_obs_buffer_to_device(self):
    for k in self.obs_buffer.keys():
      for list_element in self.obs_buffer[k]:
        #print(list_element.device)
        list_element = list_element.to(self.device)
        #print(list_element.device)

  def get_obs(self, action=None, deterministic: bool = False) -> torch.Tensor | np.ndarray:
    """
    Returns a `(batch_size, n_features)` `tensor` containing the (potientially time-delayed) observations.
    By default, this is the task goal, followed by the output of the :meth:`get_proprioception()` method, 
    the output of the :meth:`get_vision()` method, and finally the last :attr:`action_frame_stacking` action sets,
    if a non-zero `action_frame_stacking` keyword argument was passed at initialization of this class instance.
    `.i.i.d.` Gaussian noise is added to each element in the `tensor`,
    using the :attr:`obs_noise` attribute.
    """
    self.update_obs_buffer(action=action)

    obs_as_list = []
    # 1. Goal position, subtract the min fingertip position (which are negative) to make all positive
    if self.return_positive_xy_obs: 
        self.goal = self.goal.to(self.device)
        obs_as_list.append(self.goal - self.min_xy_fingertip)
    else: obs_as_list.append(self.goal)
    # 2. Fingertip position
    if self.visual_feedback:  # add fingertip position delayed
        if self.return_positive_xy_obs:
            # print(self.obs_buffer["vision"][0].device)
            # print(self.min_xy_fingertip.device)
            obs_as_list.append(self.obs_buffer["vision"][0] - self.min_xy_fingertip)
        else:
            obs_as_list.append(self.obs_buffer["vision"][0])
    # 3. Muscle lengths and velocities
    if self.return_positive_vel_obs:
        # muscle length & velocity delayed
        lengths = self.obs_buffer["proprioception"][0][:,:-6]
        vels = self.obs_buffer["proprioception"][0][:,-6:]
        obs_as_list.append(lengths)
        obs_as_list.append(torch.clamp(vels, 0, None))
        obs_as_list.append(torch.clamp(-vels, 0, None))
    else:
        obs_as_list.append(self.obs_buffer["proprioception"][0])
    # 4. Previous actions (muscle commands?) if using
    obs_as_list += self.obs_buffer["action"][:self.action_frame_stacking]

    #5. irrelevant noise inputs
    if self.irrel_noise_dims > 0:
        obs_as_list += self.irrel_inputs_buffer

    obs_as_list = [t.to(device) for t in obs_as_list]
    obs = torch.cat(obs_as_list, dim=-1).to(device) 
    if self.positive_noise and deterministic is False:
        # also draw random for each batch element
        obs += torch.tensor(np.abs(self.np_random.normal(scale=self.obs_noise, size=obs.shape))).to(self.device)
    elif deterministic is False:
        obs += torch.tensor(self.np_random.normal(scale=self.obs_noise)).to(self.device)
    obs = obs.to(device)
    return obs if self.differentiable else self.detach(obs)
  
  def update_obs_buffer(self, action=None):
    self.obs_buffer["proprioception"] = self.obs_buffer["proprioception"][1:] + [self.get_proprioception().to(device)]
    self.obs_buffer["vision"] = self.obs_buffer["vision"][1:] + [self.get_vision().to(device)]

    if action is not None:
      self.obs_buffer["action"] = self.obs_buffer["action"][1:] + [action.reshape(-1, self.action_space.shape[0])]

    self.send_obs_buffer_to_device()

    # for k in self.obs_buffer.keys():
    #    for list_element in self.obs_buffer[k]:
    #         list_element = list_element.to(device)
      
  def get_proprioception(self) -> torch.Tensor:
    """
    Returns a `(batch_size, n_features)` `tensor` containing the instantaneous (non-delayed) proprioceptive 
    feedback. By default, this is the normalized muscle length for each muscle, followed by the normalized
    muscle velocity for each muscle as well. `.i.i.d.` Gaussian noise is added to each element in the `tensor`,
    using the :attr:`proprioception_noise` attribute.
    """
    qvel = torch.split(self.states["joint"], 2, dim=1)[-1][:, None, :]
    pvel = torch.split(self.states["cartesian"], 2, dim=1)[-1][:, None, :]
    mlen = self.states["muscle"][:, 1:2, :] / self.muscle.l0_ce
    mvel = self.states["muscle"][:, 2:3, :] / self.muscle.vmax
    prop = torch.cat([mlen, mvel], dim=-1).squeeze(dim=1)
    return self.apply_noise(prop, self.proprioception_noise)


def detach_tensors_in_structure(data):
    """
    Recursively detaches all PyTorch tensors in a nested data structure.
    
    Args:
        data (dict, list, tuple, torch.Tensor, or other): The data structure to process.
    
    Returns:
        The same data structure with all tensors detached.
    """
    if isinstance(data, torch.Tensor):
        return data.clone().detach().requires_grad_(False)
    elif isinstance(data, dict):
        for key, value in data.items():
            data[key] = detach_tensors_in_structure(value)
        return data
    elif isinstance(data, list):
        return [detach_tensors_in_structure(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(detach_tensors_in_structure(item) for item in data)
    else:
        return data  # For other data types, do nothing
