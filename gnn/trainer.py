import argparse
import time
import numpy as np
import itertools
import torch
import torch.nn as nn
import torch_geometric



from envs.env import EnvManager
from ddpg import DDPG

# debug
from models.gnn import GNN_Actor, GNN_Critic
from models.vgg import VGG
# debug

torch.backends.cudnn.benchmark=True
    

def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
    
    parser.add_argument('--log_dir', default='./logs/fp', help='base directory to save logs')
    parser.add_argument('--model_dir', default='', help='base directory to save logs')
    parser.add_argument('--data_root', default='./data/processed_data', help='root directory for data')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
    parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')

    parser.add_argument('--model', default='ddpg.pth')
    parser.add_argument('--latent_vector_dim', type=int, default=128, help='')
    parser.add_argument('--state_vector_dim', type=int, default=0, help='')
    parser.add_argument('--action_space_dim', type=int, default=2, help='')
    parser.add_argument('--lra', default=1e-3, type=float)
    parser.add_argument('--lrc', default=1e-3, type=float)
    parser.add_argument('--lre', default=1e-3, type=float)
    parser.add_argument('--lrg', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--tau', default=.005, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--capacity', default=500000, type=int)
    parser.add_argument('--episode', type=int, default=2000, help='training epoch size') 
    parser.add_argument('--warmup', default=10000, type=int) 
    parser.add_argument('--cuda', default=True, action='store_true')
    parser.add_argument('--channel', type=int, default=2)
    # usv
    parser.add_argument('--usv_agent_num', default=3, type=int, help='number of usv in multi-agent')
    parser.add_argument('--usv_max_v', default=1, type=float, help='')
    parser.add_argument('--usv_collision_radius', default=2.5, type=float, help='')
    parser.add_argument('--usv_sense_radius', default=8, type=float, help='')
    # uav
    parser.add_argument('--uav_agent_num', default=0, type=int, help='number of uav in multi-agent')
    parser.add_argument('--uav_max_v', default=5, type=float, help='')
    parser.add_argument('--uav_collision_radius', default=1, type=float, help='')
    parser.add_argument('--uav_sense_radius', default=8, type=float, help='')
    # env 
    parser.add_argument('--env_dim', default=[100, 100], type=int, nargs='+', help='[width, height]')
    parser.add_argument('--debug_view_px', default=400, type=int, help='')
    parser.add_argument('--env_px', default=1000, type=int, help='')
    # parser.add_argument('--env_grid_px_per_m', default=10, type=int, help='')
    parser.add_argument('--goal_num', default=3, type=int, help='')
    parser.add_argument('--goal_margin', default=2, type=int, help='')
    parser.add_argument('--max_time_steps', default=500, type=int, help='')
    parser.add_argument('--dt', default=0.1, type=float, help='')
    
    args = parser.parse_args()
    return args

def train(args, env, agent, device):

    env.initialize()


    total_steps = 0
    ewma_reward = 0
    
    for episode in range(args.episode):
        total_reward = 0
        enable_reset = 0
        observation = env.reset()
        for step in itertools.count(start=1):
            for agent_index in range(args.usv_agent_num + args.uav_agent_num):
                if total_steps < args.warmup:
                    action_array = env.randomAction(agent_index)
                    action = action_array[0]
                else:
                    action = agent.selectAction(observation["local_fov"][agent_index])
                

                
                next_observation, reward, finished = env.step(agent_index, action)
                # (fov, state, action, reward, next_fov, next_state, finished)

                # print(type(observation["local_fov"][agent_index]))
                # print(type(observation["pos"][agent_index]))
                # print(type())
                # print(type())
                # print(type())
                # print(type())
                # print(type())
                # print(type())

                agent.append(
                    observation["local_fov"][agent_index], 
                    observation["pos"][agent_index], 
                    action, 
                    reward, 
                    next_observation["local_fov"][agent_index], 
                    next_observation["pos"][agent_index], 
                    finished
                )
                if total_steps >= args.warmup:
                    agent.update()
            
                observation = next_observation
                total_reward += reward
                total_steps += 1

                if finished:
                    enable_reset = 1
                    print(
                        'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}'
                        .format(total_steps, episode, step, total_reward))
            if enable_reset:
                break

        if episode % 200 == 0 and episode != 0:
            # print("Do the test in episode: ", episode)
            pass
            # test(args, env, agent)
    return

def debug_encoder(args, env, agent, device):
    agent = agent.to(device)

    env.initialize()
    state = env.reset()
    data = state["local_fov"][0]
    print(type(data))
    print(np.shape(data))
    data = np.array(data).reshape(1, 18, 18, args.channel)

    train_input = torch.tensor(data).to(device)
    train_input = train_input.permute(0,3,1,2)
    train_input = train_input.float()

    h = agent(train_input)
    print(type(h))
    print(h.size())
    print(h.view(1,-1).size())
    return


if __name__ == "__main__":
    arg = parse_args()

    if arg.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = 'cuda'
    else:
        device = 'cpu'
    

    # nns = generate_nn_instance(
    #     6, 32, 3, "relu"
    # )
    # gnn = ModGNNConv(nns["gnn"], aggr="add").jittable


    env = EnvManager(arg)

    ### Debug

    # agent = VGG(arg.latent_vector_dim, arg.channel)
    # debug_encoder(arg, env, agent, device)

    # agent = GNN_Critic(128, 0, 2)
    # test =  torch.randn(16, 128, 1, 1)
    # test2 = torch.randn(16, 2, 1, 1)

    # re = agent(test.reshape(16, 128), test2.reshape(16, 2))
    ### Debug


    ### Train single agent



    ### Train single agent
    
    
    ### Train

    agent = DDPG(arg)
    train(arg, env, agent, device)
    agent.save(arg.model, checkpoint=True)

    ### Train

    ### Manual

    # env.initialize()
    # unused = env.reset()
    # env.manualDisplay()

    ### Manual
