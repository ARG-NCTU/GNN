import argparse
import time
import numpy as np
import itertools
import torch
import torch.nn as nn
import torch_geometric



from envs.env import EnvManager
# from ddpg import DDPG

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


    parser.add_argument('--latent_vector_dim', type=int, default=128, help='')
    parser.add_argument('--state_vector_dim', type=int, default=4, help='')
    parser.add_argument('--action_space_dim', type=int, default=2, help='')
    parser.add_argument('--lra', default=1e-3, type=float)
    parser.add_argument('--lrc', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--tau', default=.005, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--capacity', default=500000, type=int)
    parser.add_argument('--episode', type=int, default=2000, help='training epoch size') 
    parser.add_argument('--cuda', default=True, action='store_true')
    # usv
    parser.add_argument('--usv_agent_num', default=3, type=int, help='number of usv in multi-agent')
    parser.add_argument('--usv_max_v', default=1.5, type=float, help='')
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

def train(args, env, agent):

    total_steps = 0
    ewma_reward = 0
    for episode in range(args.episode):
        # total_reward = 0
        # env.initialize()
        # state = env.fetchState()
        # for step in itertools.count(start=1):
        #     if step < args.warmup:
        #         action = env.randomAction()
        #     else:
        #         action = agent.select_action(state, noise=False)
            
        #     next_state, reward, done = env.step(action)
        #     agent.append(state, action, reward, next_state, done)
        #     if total_steps >= args.warmup:
        #         agent.update()
            
        #     state = next_state
        #     total_reward += reward
        #     total_steps += 1
        #     if done:
        #         ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
        #         writer.add_scalar('Train/Episode Reward', total_reward,
        #                           total_steps)
        #         writer.add_scalar('Train/Ewma Reward', ewma_reward,
        #                           total_steps)
        #         print(
        #             'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}'
        #             .format(total_steps, episode, t, total_reward,
        #                     ewma_reward))
        #         break

        if episode % 200 == 0 and episode != 0:
            print("Do the test in episode: ", episode)
            # pass
            # test(args, env, agent)



    # # old version
    # env.initialize()
    # env.agentPerceptionUpdate(0)
    # data = env.getAgentFOV(0)

    # # data = np.zeros((10,64,64,3))
    
    # print(type(data))
    # print(np.shape(data))
    # data = np.array(data).reshape(1, 18, 18, 3)
    # # data = np.array(data).reshape(1, 64, 64, 3)
    # train_input = torch.tensor(data).to(device)
    # train_input = train_input.permute(0,3,1,2)
    # train_input = train_input.float()


    # h = module["encoder"](train_input)

    # print(np.shape(h))

    return


if __name__ == "__main__":
    arg = parse_args()

    
    
    env = EnvManager(arg)

    
    # agent = DDPG(arg)

    
    # train(arg, env, agent)

    # nns = generate_nn_instance(
    #     6, 32, 3, "relu"
    # )
    # gnn = ModGNNConv(nns["gnn"], aggr="add").jittable
    
    
    env.initialize()

    env.manualDisplay()
