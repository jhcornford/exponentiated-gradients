import torch
import matplotlib.pyplot as plt
import numpy as np


use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

def tensor(x): return x if torch.is_tensor(x) else torch.tensor(x).to(device).float()
def detach(x): return x.clone().detach().cpu().numpy()

def eval_and_plot(controller):
    from train_controller import forward_pass 
    viz_batch_i = 0
    batch_size = 8
    fig, axs = plt.subplots(2, 5,)
    fig.set_size_inches((15, 6))
    fig.set_dpi(200)
    #fig.suptitle(f"Network evaluation epoch-{epoch}", fontsize=10)
    
    for i, eval_mode in enumerate([False, True]):
        outputs = forward_pass(controller, batch_size=batch_size, n_timesteps=100, eval_mode=eval_mode)
        env_fingertip = np.stack([detach(obs) for obs in outputs["env_fingertip"]])
        targets = np.stack([detach(obs) for obs in outputs["env_goal"]])

        x = env_fingertip[:, :, 0]
        y = env_fingertip[:, :, 1]
        axs[i,0].plot(x, y)

        x = targets[-1, :, 0] # -1 for last timestep
        y = targets[-1, :, 1]
        axs[i,0].scatter(x, y)
        axs[i,0].set_title(f"Target reaches")
        axs[i,0].set_xlabel("X position (m)")
        axs[i,0].set_ylabel("Y position (m)")

        hidden_states = np.stack([detach(obs) for obs in outputs["policy_states"]])
        hidden_states = hidden_states.squeeze() # (n_timesteps, batch_size, hidden_dim)
        axs[i,1].plot(hidden_states[:, viz_batch_i, :])
        axs[i,1].set_title(f"Hidden states (target {viz_batch_i})")
        axs[i,1].set_xlabel("Timestep")

        actions = np.stack([detach(o) for o in outputs["actions"]]).squeeze()
        xy = np.stack(actions)[:, viz_batch_i, :]
        axs[i,2].plot(xy)
        axs[i,2].set_title(f"Muscle drive (target {viz_batch_i})")
        axs[i,2].set_xlabel("Timestep")

        obs = torch.stack(outputs["observations"]).detach().cpu().numpy()

        axs[i,3].plot(obs[:,viz_batch_i,:22]);
        axs[i,3].set_title(f"True muscle and joint feedback")
        axs[i,3].set_ylabel("Activation (a.u.)")
        axs[i,3].set_xlabel("Time (ms)")

        axs[i,4].plot(obs[:,viz_batch_i,-22:]);
        axs[i,4].set_title(f"Noise feedback")
        axs[i,4].set_ylabel("Activation (a.u.)")
        axs[i,4].set_xlabel("Time (ms)")

    plt.tight_layout()

    return fig

def plot_losses(losses, weight_norms, gradient_norms):
    fig, axs = plt.subplots(1, 3,)
    fig.set_size_inches((10, 3))
    fig.set_dpi(200)

    axs[0].plot(np.stack(losses))
    axs[0].set_title(f"Loss")

    axs[1].plot(np.stack(weight_norms))
    axs[1].set_title(f"Parameter norms")

    axs[2].plot(np.stack(gradient_norms))
    axs[1].set_title(f"Gradient norms")

    axs[0].set_xlabel("Iteration #")
    axs[1].set_xlabel("Iteration #")
    axs[2].set_xlabel("Iteration #")

    plt.tight_layout()
    return fig
