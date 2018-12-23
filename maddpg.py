# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
import numpy as np
from utilities import soft_update, transpose_to_tensor, transpose_list
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

class MADDPG:
    def __init__(self, episodes_before_train, batch_size, replay_buffer, discount_factor=0.95, tau=0.02):
        super(MADDPG, self).__init__()

        # critic input = obs_full + actions = 24+24+2+2=52
        self.maddpg_agent = [DDPGAgent(24, 2, 400, 300, 48, 4, 400, 300),
                             DDPGAgent(24, 2, 400, 300, 48, 4, 400, 300)]

        
        self.num_agents = 2
        self.action_size = 2
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0
        self.episodes_before_train = episodes_before_train
        self.batch_size = batch_size
        self.buffer =  replay_buffer
        

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return np.array(actions)

    def target_act(self, obs, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        #return target_actions
        target_actions = torch.zeros(obs.shape[:2] + (self.action_size,), dtype=torch.float, device=device)
        for i in range(self.num_agents):
            target_actions[:, i, :] = self.maddpg_agent[i].target_act(obs[:, i])
        
        return target_actions

    def update(self, samples, agent_number):
        """update the critics and actors of all the agents """

        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]
        obs, obs_full, action, reward, next_obs, next_obs_full, done = samples
        
        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_obs)
        
        with torch.no_grad():
            q_next = agent.target_critic(next_obs_full, target_actions.view(-1, 4))

        y = reward[:,agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[:, agent_number].view(-1, 1))
        q = agent.critic(obs_full, action.view(-1, 4))

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        
        agent_obs = obs[:, agent_number]
        agent_actions = agent.actor(agent_obs)
        q_input = action.clone()
        q_input[:, agent_number] = agent_actions

        # get the policy gradient
        actor_loss = -agent.critic(obs_full, q_input.view(-1, 4)).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        
        return al, cl
    
    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
    
    def reset(self):
        for ddpg_agent in self.maddpg_agent:
            ddpg_agent.reset()          
    
    def to_tensor(self, samples):
        obs, obs_full, actions, rewards, next_states, next_states_full, dones = samples

        obs = torch.from_numpy(obs).float().to(device)
        obs_full = torch.from_numpy(obs_full).float().to(device)
        actions = torch.from_numpy(actions).float().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        next_states_full = torch.from_numpy(next_states_full).float().to(device)
        dones = torch.from_numpy(dones.astype(np.uint8)).float().to(device)

        return obs, obs_full, actions, rewards, next_states, next_states_full, dones
    
    def step(self, i_episode, obs, actions, rewards, next_states, dones):
        obs_full = obs.reshape(-1)
        next_states_full = next_states.reshape(-1)
        
        self.buffer.add(obs=obs, obs_full=obs_full, action=actions, reward=rewards,
                        next_state=next_states, next_state_full=next_states_full, done=dones)

        self.i_episode = i_episode
        if (i_episode >= self.episodes_before_train) and (self.buffer.size() >= self.batch_size):
            if (self.i_episode == self.episodes_before_train) and np.any(dones):
                print("\nStart training...")

            for agent_i in range(self.num_agents):
                samples = self.buffer.sample(self.batch_size)
                self.update(self.to_tensor(samples), agent_i)
            self.update_targets()
