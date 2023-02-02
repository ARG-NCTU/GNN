import argparse
from collections import deque
import time
import numpy as np
import torch
import torch.nn as nn
import torch_geometric
import torch.optim as optim
import random



from models.gnn import GNN_Actor, GNN_Critic
from models.vgg import VGG

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
        # (fov, state, action, reward, next_fov, next_state, finished)
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

        # encoder network
        self.encoder_net = VGG(self.cfg.latent_vector_dim, self.cfg.channel).to(device)

        # behavior network
        self.actor_net = GNN_Actor(self.cfg.latent_vector_dim, self.cfg.state_vector_dim, self.cfg.action_space_dim).to(device)
        self.critic_net = GNN_Critic(self.cfg.latent_vector_dim, self.cfg.state_vector_dim, self.cfg.action_space_dim).to(device)

        # target network
        self.target_actor_net = GNN_Actor(self.cfg.latent_vector_dim, self.cfg.state_vector_dim, self.cfg.action_space_dim).to(device)
        self.target_critic_net = GNN_Critic(self.cfg.latent_vector_dim, self.cfg.state_vector_dim, self.cfg.action_space_dim).to(device)

        # initialize target network
        self.target_actor_net.load_state_dict(self.actor_net.state_dict())
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())

        # optomizer
        self.actor_opt = optim.Adam(self.actor_net.parameters(), lr=self.cfg.lra)
        self.critic_opt = optim.Adam(self.critic_net.parameters(), lr=self.cfg.lrc)
        self.encoder_opt = optim.Adam(self.encoder_net.parameters(), lr=self.cfg.lre)

        # action noise
        self.action_noise = GaussianNoise(dim=2)

        # memory
        self.memory = ReplayMemory(capacity=self.cfg.capacity)

        

    def selectAction(self, fov, noise=True):
        # encoder_input = state["local_fov"][agent_index]
        encoder_input = np.array(fov).reshape(1, 18, 18, self.cfg.channel)
        encoder_input = torch.tensor(encoder_input).to(self.device)
        encoder_input = encoder_input.permute(0,3,1,2)
        encoder_input = encoder_input.float()

        with torch.no_grad():
            h = self.encoder_net(encoder_input)
            if noise:
                re = self.actor_net(h.view(1,-1).to(self.device))+\
                       torch.from_numpy(self.action_noise.sample()).view(1,-1).to(self.device)
            else:
                re = self.actor_net(h.view(1,-1).to(self.device))
        return re.cpu().numpy().squeeze()


    def append(self, fov, state, action, reward, next_fov, next_state, finished):
        # (fov, state, action, reward, next_fov, next_state, finished)
        self.memory.append(
            fov,
            state, 
            action, 
            [reward], 
            next_fov,
            next_state,
            [int(finished)]
            )

    def update(self):
        # update the behavior networks
        self.updateBehaviorNetwork(self.gamma)
        # update the target networks
        self.updateTargetNetwork(self.target_actor_net, self.actor_net, self.tau)
        self.updateTargetNetwork(self.target_critic_net, self.critic_net, self.tau)

    def updateBehaviorNetwork(self, gamma):
        actor_net, critic_net, encoder_net, target_actor_net, target_critic_net = self.actor_net, self.critic_net, self.encoder_net, self.target_actor_net, self.target_critic_net
        actor_opt, critic_opt, encoder_opt = self.actor_opt, self.critic_opt, self.encoder_opt

        # sample a minibatch of transitions
        fov, state, action, reward, next_fov, next_state, finished = self.memory.sample(
            self.batch_size, self.device)

        ## update critic ##

        encoder_input = np.array(fov.cpu().detach().numpy())
        encoder_input = torch.tensor(encoder_input).to(self.device)
        encoder_input = encoder_input.permute(0,3,1,2)
        encoder_input = encoder_input.float()
        h = self.encoder_net(encoder_input)
        h = h.reshape(self.cfg.batch_size, self.cfg.latent_vector_dim)
        

        q_value = self.critic_net(h, action)
        with torch.no_grad():
            next_encoder_input = np.array(next_fov.cpu().detach().numpy())
            next_encoder_input = torch.tensor(next_encoder_input).to(self.device)
            next_encoder_input = next_encoder_input.permute(0,3,1,2)
            next_encoder_input = next_encoder_input.float()
            next_h = self.encoder_net(next_encoder_input)
            next_h = next_h.reshape(self.cfg.batch_size, self.cfg.latent_vector_dim)

            a_next = self.target_actor_net(next_h)
            q_next = self.target_critic_net(next_h, a_next)
            q_target = reward + gamma * q_next * (1-finished)
        criterion = nn.MSELoss()
        critic_loss = criterion(q_value, q_target)


        # optimize critic
        actor_net.zero_grad()
        critic_net.zero_grad()
        encoder_net.zero_grad()
        critic_loss.backward(retain_graph=True)
        critic_opt.step()

        ## update actor ##

        action = self.actor_net(h)
        actor_loss = -self.critic_net(h, action).mean()

        # optimize actor & # encoder
        actor_net.zero_grad()
        critic_net.zero_grad()
        encoder_net.zero_grad()
        actor_loss.backward()
        actor_opt.step()
        encoder_opt.step()
        
        
        
        


    def updateTargetNetwork(self, target_net, net, tau):
        for target, behavior in zip(target_net.parameters(), net.parameters()):
            target.data.copy_((1-tau)*target.data + tau*behavior.data)

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save(
                {
                    'actor': self.actor_net.state_dict(),
                    'critic': self.critic_net.state_dict(),
                    'encoder': self.encoder_net.state_dict(),
                    'target_actor': self.target_actor_net.state_dict(),
                    'target_critic': self.target_critic_net.state_dict(),
                    'actor_opt': self.actor_opt.state_dict(),
                    'critic_opt': self.critic_opt.state_dict(),
                    'encoder_opt': self.encoder_opt.state_dict(),
                }, model_path)
        else:
            torch.save(
                {
                    'actor': self._actor_net.state_dict(),
                    'critic': self._critic_net.state_dict(),
                    'encoder': self.encoder_net.state_dict(),
                }, model_path)

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path)
        self.actor_net.load_state_dict(model['actor'])
        self.critic_net.load_state_dict(model['critic'])
        self.encoder_net.load_state_dict(model['encoder'])
        if checkpoint:
            self.target_actor_net.load_state_dict(model['target_actor'])
            self.target_critic_net.load_state_dict(model['target_critic'])
            self.actor_opt.load_state_dict(model['actor_opt'])
            self.critic_opt.load_state_dict(model['critic_opt'])
            self.encoder_opt.load_state_dict(model['encoder_opt'])
