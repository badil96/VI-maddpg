import argparse
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
from pettingzoo.mpe import simple_adversary_v2, simple_spread_v2, simple_tag_v2
from pettingzoo.classic import rps_v2
import torch
import copy
from PIL import Image

import mp_v0

from MADDPG import MADDPG

from agent_test import agent_test

# def agent_test(maddpg): #RPS
#     prop_dist = {}
#     test_env = mp_v0.parallel_env()
    
#     for agent_id, agent in maddpg.agents.items():
#         probs = []
#         # for i in range(1500):
#         obs = np.array(np.random.choice([0,1,2], 1500))
#         with torch.no_grad():
#             o = torch.from_numpy(obs).unsqueeze(1).float()
#             _, logits = agent.action(o, model_out=True)
#             p = torch.nn.functional.softmax(logits, dim=-1)
#                 # probs.append(p)
#         # mean = torch.mean(torch.stack(probs), 0)
#         # x = mean - torch.tensor([0.3333, 0.3333, 0.3333])
#         prop_dist[agent_id] =  p
#     return prop_dist

# test_seeds = [ 530,  527,  538,  561,  141, 1217,  974, 1168,  964, 1061,  791,
#         639,  335,  464,  398,  958,  792,  616,   39,  262, 1199,  376,
#        1318,  609,  153,  714, 1153,  221,  449,  602,   30,  951, 1067,
#        1073,  293,  911, 1474,  923,  993, 1031,  838, 1474,  434, 1269,
#        1037,  190, 1403, 1364,   77,  391,  827,  377,  246,  765,  742,
#         286, 1328,  547, 1490, 1439,  918, 1201,  125, 1181,  193,  805,
#         224,  139,  877,  531,  421,  182,  383, 1336,  368,  303,  990,
#         238,   16,  938, 1279,  305,  221,  931, 1323, 1365, 1021,  959,
#         138,  254,  739, 1130, 1120, 1416,  258,  460,  370, 1317,   32,
#          27]
# def agent_test(agent, episode, steps, results_dir): #simple adversary
#     gif_dir = os.path.join(results_dir, 'gif_test')
#     if not os.path.exists(gif_dir):
#         os.makedirs(gif_dir)

#     test_env = simple_adversary_v2.parallel_env( max_cycles=25)
#     states = test_env.reset()
#     agent_rewards = {agent: [] for agent in test_env.agents}
#     # frame_list = []  # used to save gif
#     for i in test_seeds:
#         states = test_env.reset(seed=int(i))
#         # reward of each episode of each agent
#         while test_env.agents:  # interact with the env for an episode
#             actions = agent.select_action(states)
#             next_states, rewards, dones, infos = test_env.step(actions)       
#             states = next_states
#         # frame_list.append(Image.fromarray(test_env.render(mode='rgb_array')))
#         for agent_id, r in rewards.items():  # record reward
#                 agent_rewards[agent_id].append(r) 


#         test_env.close()
#     # save gif
#     # frame_list[0].save(os.path.join(gif_dir, f'out{episode}.gif'),
#     #                  save_all=True, append_images=frame_list[1:], duration=1, loop=0)
#     return agent_rewards

# def agent_test(agent, episode, steps, results_dir): #simple tag
#   gif_dir = os.path.join(results_dir, 'gif_test')
#   if not os.path.exists(gif_dir):
#       os.makedirs(gif_dir)

#   test_env = simple_tag_v2.parallel_env(num_good=1, num_adversaries=2, num_obstacles=2, max_cycles=25)
#   states = test_env.reset()
#   agent_rewards = {agent: [0]*100 for agent in test_env.agents}
#   agent_distances = {agent: [[] for _ in range(100)] for agent in test_env.agents}
#   # frame_list = []  # used to save gif
#   for i in range(len(test_seeds)):
#     states = test_env.reset(seed=int(test_seeds[i]))
#     while test_env.agents:  # interact with the env for an episode
#         actions = agent.select_action(states)
#         next_states, rewards, dones, infos = test_env.step(actions)
#         # frame_list.append(Image.fromarray(test_env.render(mode='rgb_array')))
#         states = next_states
#         for agent_id, r in rewards.items():  # record reward
#             agent_rewards[agent_id][i] += r 
#         for agent_id, obs in states.items():
#             agent_distances[agent_id][i].append([obs[2],obs[3]])

#   test_env.close()
#   # save gif
#   # frame_list[0].save(os.path.join(gif_dir, f'out{episode}.gif'),
#   #                    save_all=True, append_images=frame_list[1:], duration=1, loop=0)
#   return agent_rewards, agent_distances

    
def lookahead_avg(param, param_k, alpha=0.5):
  with torch.no_grad(): return param + alpha * (param_k - param)

def lookahead(old_actor, new_actor):
  for (old_name, old_param), (new_name, new_param) in zip(old_actor.named_parameters(), new_actor.named_parameters()):
    new_param.data = lookahead_avg(old_param.data, new_param.data)


def get_env(env_name, ep_len=25):
    """create environment and get observation and action dimension of each agent in this environment"""
    new_env = None
    if env_name == 'simple_adversary_v2':
        new_env = simple_adversary_v2.parallel_env(max_cycles=ep_len)
    if env_name == 'simple_spread_v2':
        new_env = simple_spread_v2.parallel_env(max_cycles=ep_len)
    if env_name == 'simple_tag_v2':
        new_env = simple_tag_v2.parallel_env(num_good=1, num_adversaries=2, num_obstacles=2, max_cycles=ep_len)
    if env_name == 'rps_v2':
        new_env = rps_v2.parallel_env(max_cycles=ep_len)
    if env_name == 'pd_v0':
        new_env = pd_v0.parallel_env()
    if env_name == 'mp_v0':
        new_env = mp_v0.parallel_env()

    new_env.reset()
    _dim_info = {}
    for agent_id in new_env.agents:
        if env_name in ['rps_v2', 'pd_v0', 'mp_v0']:
            _dim_info[agent_id] = []  # [obs_dim, act_dim]
            _dim_info[agent_id].append(1)
            _dim_info[agent_id].append(new_env.action_space(agent_id).n)
        else:
            _dim_info[agent_id] = []  # [obs_dim, act_dim]
            _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])
            _dim_info[agent_id].append(new_env.action_space(agent_id).n)

    return new_env, _dim_info

def get_running_reward(arr: np.ndarray, window=100):
            """calculate the running reward, i.e. average of last `window` elements from rewards"""
            running_reward = np.zeros_like(arr)
            for i in range(window - 1):
                running_reward[i] = np.mean(arr[:i + 1])
            for i in range(window - 1, len(arr)):
                running_reward[i] = np.mean(arr[i - window + 1:i + 1])
            return running_reward



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str, default='simple_adversary_v2', help='name of the env',
                        choices=['simple_adversary_v2', 'simple_spread_v2', 'simple_tag_v2', 'rps_v2','pd_v0','mp_v0'])
    parser.add_argument('--episode_num', type=int, default=30000,
                        help='total episode num during training procedure')
    parser.add_argument('--episode_length', type=int, default=25, help='steps per episode')
    parser.add_argument('--learn_interval', type=int, default=100,
                        help='steps interval between learning time')
    parser.add_argument('--random_steps', type=int, default=1024,
                        help='random steps before the agent start to learn')
    parser.add_argument('--tau', type=float, default=0.02, help='soft update parameter')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--buffer_capacity', type=int, default=int(1e6), help='capacity of replay buffer')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch-size of replay buffer')
    parser.add_argument('--actor_lr', type=float, default=0.01, help='learning rate of actor')
    parser.add_argument('--critic_lr', type=float, default=0.01, help='learning rate of critic')
    args = parser.parse_args()

    


    seeds = [ 53]
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        # create folder to save result
        env_dir = os.path.join('./results', args.env_name)
        if not os.path.exists(env_dir):
            os.makedirs(env_dir)
        total_files = len([file for file in os.listdir(env_dir)])
        result_dir = os.path.join(env_dir, 'seed'+f'{seed}')
        os.makedirs(result_dir)

        env, dim_info = get_env(args.env_name, args.episode_length)
        maddpg = MADDPG(dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr,
                        result_dir)


        step = 0  # global step counter
        agent_num = env.num_agents
        # reward of each episode of each agent
        episode_rewards = {agent_id: np.zeros(args.episode_num) for agent_id in env.agents}
        
        test_scores_dict = {}
        test_distances_dict = {}
            
        # LA
        la_step = 40
        la_ss_step = 400
        la_sss_step = 4000
        la_actors = {}
        la_ss_actors = {}
        la_sss_actors = {}
        la_step_critic = 40
        la_ss_step_critic = 400
        la_sss_step_critic = 4000
        la_critics = {}
        la_ss_critics = {}
        la_sss_critics = {}
        for agent_id, agent in maddpg.agents.items():
          #append every actor params to the list
          la_actors[agent_id] = copy.deepcopy(agent.actor)
          la_ss_actors[agent_id] = copy.deepcopy(agent.actor)
          la_sss_actors[agent_id] = copy.deepcopy(agent.actor)

        #   append every critic params to the list
          la_critics[agent_id] = copy.deepcopy(agent.critic)
          la_ss_critics[agent_id] = copy.deepcopy(agent.critic)
          la_sss_critics[agent_id] = copy.deepcopy(agent.critic)


        for episode in range(args.episode_num):
            obs = env.reset()
            agent_reward = {agent_id: 0 for agent_id in env.agents}  # agent reward of the current episode
            while env.agents:  # interact with the env for an episode
                step += 1
                if step < args.random_steps:
                    action = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
                else:
                    action = maddpg.select_action(obs)

                next_obs, reward, done, info = env.step(action)
                # env.render()
                maddpg.add(obs, action, reward, next_obs, done)

                for agent_id, r in reward.items():  # update reward
                    agent_reward[agent_id] += r

                if step >= args.random_steps and step % args.learn_interval == 0:  # learn every few steps
                    if maddpg.buffer_size > args.batch_size:
                        maddpg.learn(args.batch_size, args.gamma)
                        maddpg.update_target(args.tau)
                        # for agent_id, agent in maddpg.agents.items():
                        #     agent.actor_scheduler.step()
                        #     agent.critic_scheduler.step()

                obs = next_obs

            # episode finishes
            for agent_id, r in agent_reward.items():  # record reward
                episode_rewards[agent_id][episode] = r

            if (episode + 1) % 100 == 0:  # print info every 100 episodes
                message = f'episode {episode + 1}, '
                sum_reward = 0
                for agent_id, r in agent_reward.items():  # record reward
                    message += f'{agent_id}: {r:>4f}; '
                    sum_reward += r
                message += f'sum reward: {sum_reward}'
                print(message)
                # print('actor_lr:', agent.actor_scheduler.get_last_lr()) 
                # print('critic_lr:', agent.critic_scheduler.get_last_lr()) 

            # # Perform  lookahead step according to LA_STEP
            # if ((episode + 1) % la_step == 0):
            #   for agent_id, agent in maddpg.agents.items():
            #       #actor
            #       lookahead(la_actors[agent_id], agent.actor)
            #       #la_actor = copy.deepcopy(agent.actor)
            #       la_actors[agent_id] = copy.deepcopy(agent.actor)
              

            # if ((episode + 1) % la_step_critic == 0):
            #   for agent_id, agent in maddpg.agents.items():      
            #       #critic
            #       lookahead(la_critics[agent_id], agent.critic)
            #       #la_critic = copy.deepcopy(agent.critic)
            #       la_critics[agent_id] = copy.deepcopy(agent.critic)

            # # # #update target networks    
            # # #   maddpg.update_target(args.tau)
                          
            # # Perform nested lookahead step according to LA_SS_STEP
            # if ((episode + 1) % la_ss_step == 0):
            #   for agent_id, agent in maddpg.agents.items():
            #       #actor
            #       lookahead(la_ss_actors[agent_id], agent.actor)
            #       # update both la and nested la copies
            #       #la_actor = copy.deepcopy(agent.actor)
            #       la_actors[agent_id] = copy.deepcopy(agent.actor)
            #       la_ss_actors[agent_id] = copy.deepcopy(agent.actor)

                  
            # if ((episode + 1) % la_ss_step_critic == 0):
            #   for agent_id, agent in maddpg.agents.items():
            #       #critic
            #       lookahead(la_ss_critics[agent_id], agent.critic)
            #       #la_critic = copy.deepcopy(agent.critic)
            #       la_critics[agent_id] = copy.deepcopy(agent.critic)
            #       la_ss_critics[agent_id] = copy.deepcopy(agent.critic)

            # # # #     #update target networks    
            # # # #   maddpg.update_target(args.tau)

            #  # Perform 2nd nested lookahead step according to LA_SS_STEP
            # if ((episode + 1) % la_sss_step == 0):
            #   for agent_id, agent in maddpg.agents.items():
            #       #actor
            #       lookahead(la_sss_actors[agent_id], agent.actor)
            #       # update all la and nested la copies
            #       #la_actor = copy.deepcopy(agent.actor)
            #       la_actors[agent_id] = copy.deepcopy(agent.actor)
            #       la_ss_actors[agent_id] = copy.deepcopy(agent.actor)
            #       la_sss_actors[agent_id] = copy.deepcopy(agent.actor)

                  
            # if ((episode + 1) % la_sss_step_critic == 0):
            #   for agent_id, agent in maddpg.agents.items():
            #       #critic
            #       lookahead(la_sss_critics[agent_id], agent.critic)
            #       #la_critic = copy.deepcopy(agent.critic)
            #       la_critics[agent_id] = copy.deepcopy(agent.critic)
            #       la_ss_critics[agent_id] = copy.deepcopy(agent.critic)
            #       la_sss_critics[agent_id] =copy.deepcopy(agent.critic)

            # if ((episode + 1) % 20000 == 0):
            #     maddpg.clear()
            #     print('cleared the buffer')
                  
            if ((episode + 1) % 100 == 0):
                if args.env_name=='simple_tag_v2':
                    test_scores_dict[episode], test_distances_dict[episode] = agent_test(maddpg,args.env_name, episode,  results_dir =result_dir, steps = 25, save_gifs=True)
                else:
                    test_scores_dict[episode] = agent_test(maddpg, args.env_name, episode,  results_dir =result_dir, steps = 25, save_gifs=False)   

            
        with open(os.path.join(result_dir, 'rewards_seed'+f'{seed}'+'.pkl'), 'wb') as f:  # save testing data
                pickle.dump(test_scores_dict, f)
        with open(os.path.join(result_dir, 'distances_seed'+f'{seed}'+'.pkl'), 'wb') as f:  # save testing data
                pickle.dump(test_distances_dict, f)
        maddpg.save(episode_rewards)  # save model


        # training finishes, plot reward
        fig, ax = plt.subplots()
        x = range(1, args.episode_num + 1)
        for agent_id, reward in episode_rewards.items():
            ax.plot(x, reward, label=agent_id)
            ax.plot(x, get_running_reward(reward))
        ax.legend()
        ax.set_xlabel('episode')
        ax.set_ylabel('reward')
        title = f'training result of maddpg solve {args.env_name}'
        ax.set_title(title)
        plt.savefig(os.path.join(result_dir, title))