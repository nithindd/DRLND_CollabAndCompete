# individual network settings for each actor + critic pair
# see networkforall for details

from model import Actor, Critic
from utilities import hard_update, gumbel_softmax, onehot_from_logits
from torch.optim import Adam
import torch
import numpy as np


# add OU noise for exploration
from OUNoise import OUNoise

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

TAU = 1e-3  

class DDPGAgent:
    def __init__(self, in_actor, out_actor, hidden_in_actor, hidden_out_actor, state_dim_in_critic, action_dim_inp_critic, hidden_in_critic, hidden_out_critic, lr_actor=1.0e-4, lr_critic=1.0e-3):
        super(DDPGAgent, self).__init__()

        self.actor = Actor(in_actor, out_actor, hidden_in_actor, hidden_out_actor).to(device)
        self.critic = Critic(state_dim_in_critic, action_dim_inp_critic, hidden_in_critic, hidden_out_critic).to(device)
        self.target_actor = Actor(in_actor, out_actor, hidden_in_actor, hidden_out_actor).to(device)
        self.target_critic = Critic(state_dim_in_critic, action_dim_inp_critic, hidden_in_critic, hidden_out_critic).to(device)

      
        self.noise = OUNoise(out_actor, scale=1.0 )
        
        self.tau = TAU
        
        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=1.e-5)
        
    def act(self, obs, noise=0.0):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(obs).float().to(device).view(-1, 24)
        self.actor.eval()
        
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        self.actor.train()
        add_noise = noise * self.noise.noise()
        action += add_noise.cpu().data.numpy()
        
        return np.clip(action, -1, 1).reshape(-1)

    def target_act(self, obs, noise=0.0):
        obs = obs.to(device)
        action = self.target_actor(obs) + noise*self.noise.noise()
        return action
    
    def reset(self):
        self.noise.reset()
        
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def soft_update_all(self):
        DDPGAgent.soft_update(local_model=self.critic, target_model=self.critic_target, tau=self.tau)
        DDPGAgent.soft_update(local_model=self.actor, target_model=self.actor_target, tau=self.tau)