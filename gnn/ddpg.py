import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch_geometric
import torch.optim as optim



from models.gnn import GNN

class GaussianNoise:
    def __init__(self, dim, mu=None, std=None):
        self.mu = mu if mu else np.zeros(dim)
        self.std = std if std else np.ones(dim) * .1

    def sample(self):
        return np.random.normal(self.mu, self.std)


class ReplayMemory:
    __slots__ = ['buffer']

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition):
        # (state, action, reward, next_state, done)
        self.buffer.append(tuple(map(tuple, transition)))

    def sample(self, batch_size, device):
        transitions = random.sample(self.buffer, batch_size)

        return (torch.tensor(x, dtype=torch.float, device=device)
                for x in zip(*transitions))

class DDPG:
    def __init__(self, args):
        self.cfg = args

        if torch.cuda.is_available() and self.cfg.cuda == True:
            print("Device: GPU")
            device = 'cuda'
        else:
            print("Device: CPU")
            device = 'cpu'

        ## config ##
        self.device = device
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.gamma = args.gamma

        

        # behavior network
        self.actor_net = GNN(self.cfg.latent_vector_dim, self.cfg.state_vector_dim, self.cfg.action_space_dim).to(device)
        self.critic_net = GNN(self.cfg.latent_vector_dim, self.cfg.state_vector_dim, 1).to(device)

        # target network
        self.target_actor_net = GNN(self.cfg.latent_vector_dim, self.cfg.state_vector_dim, self.cfg.action_space_dim).to(device)
        self.target_critic_net = GNN(self.cfg.latent_vector_dim, self.cfg.state_vector_dim, 1).to(device)

        # initialize target network
        self.target_actor_net.load_state_dict(self.actor_net.state_dict())
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())

        # optomizer
        self.actor_opt = optim.Adam(self.actor_net.parameters(), lr=self.cfg.lra)
        self.critic_opt = optim.Adam(self.critic_net.parameters(), lr=self.cfg.lrc)

        # action noise
        self.action_noise = GaussianNoise(dim=2)

        # memory
        self.memory = ReplayMemory(capacity=self.cfg.capacity)

        

    def select_action(self, state, noise=False):
        with torch.no_grad():
            if noise:
                pass
                # re = self.actor_net(torch.from_numpy(state).view(1,-1).to(self.device))+\
                #        torch.from_numpy(self.action_noise.sample()).view(1,-1).to(self.device)
            else:
                re = self.actor_net(torch.from_numpy(state).view(1,-1).to(self.device))
        return re.cpu().numpy().squeeze()


    def append(self, state, action, reward, next_state, finished):
        self.memory.append(
            state, 
            action, 
            reward, 
            next_state,
            [int(finished)]
            )

    def update(self):
        # update the behavior networks
        self.update_behavior_network(self.gamma)
        # update the target networks
        self.update_target_network(self.target_actor_net, self.actor_net,
                                    self.tau)
        self.update_target_network(self.target_critic_net, self.critic_net,
                                    self.tau)

    def update_behavior_network(self, gamma):
        actor_net, critic_net, target_actor_net, target_critic_net = self.actor_net, self.critic_net, self.target_actor_net, self.target_critic_net
        actor_opt, critic_opt = self.actor_opt, self.critic_opt

        # sample a minibatch of transitions
        state, action, reward, next_state, finished = self.memory.sample(
            self.batch_size, self.device)

        ## update critic ##
        q_value = self.critic_net(state)
        with torch.no_grad():
           a_next = self.target_actor_net(next_state)
           q_next = self.target_critic_net(next_state)
           q_target = reward + gamma * q_next * (1-done)
        criterion = nn.MSELoss()
        critic_loss = criterion(q_value, q_target)


        # optimize critic
        actor_net.zero_grad()
        critic_net.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        ## update actor ##
        action = self.actor_net(state)
        actor_loss = -self.critic_net(state).mean()

        # optimize actor
        actor_net.zero_grad()
        critic_net.zero_grad()
        actor_loss.backward()
        actor_opt.step()


    def update_target_network(target_net, net, tau):
        '''update target network by _soft_ copying from behavior network'''
        for target, behavior in zip(target_net.parameters(), net.parameters()):
            #
            target.data.copy_((1-tau)*target.data + tau*behavior.data)
            #

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save(
                {
                    'actor': self._actor_net.state_dict(),
                    'critic': self._critic_net.state_dict(),
                    'target_actor': self._target_actor_net.state_dict(),
                    'target_critic': self._target_critic_net.state_dict(),
                    'actor_opt': self.actor_opt.state_dict(),
                    'critic_opt': self.critic_opt.state_dict(),
                }, model_path)
        else:
            torch.save(
                {
                    'actor': self._actor_net.state_dict(),
                    'critic': self._critic_net.state_dict(),
                }, model_path)

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path)
        self._actor_net.load_state_dict(model['actor'])
        self._critic_net.load_state_dict(model['critic'])
        if checkpoint:
            self._target_actor_net.load_state_dict(model['target_actor'])
            self._target_critic_net.load_state_dict(model['target_critic'])
            self.actor_opt.load_state_dict(model['actor_opt'])
            self.critic_opt.load_state_dict(model['critic_opt'])
