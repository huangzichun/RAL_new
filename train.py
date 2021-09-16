# 模型训练部分

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import random
import os
import matplotlib as mpl
import matplotlib.pyplot as plt


def train(model, data, agent_1, agent_2, state, epoch, env, budget, MEMORY_CAPACITY_1, MEMORY_CAPACITY_2):
    # single agent
    target_net_1 = torch.load('\model\model_cluster_20_v7.pkl')
    uncertain_feature_list = []
    action_value_lsit = []
    dist_sum_list = []
    for i_episode in range(epoch):
        print('epoch:', i_episode + 1)
        data.reset()
        first_num = 800
        data.first_random(first_num)
        model.train()
        model.give_label()
        acc_change = model.acc_change()
        print(acc_change)
        # if acc_change < -0.3:
        #     print('break!')
        #     os.system('pause')
        s_1 = state.update()
        for j in range(budget):
            # print('now_state1:', s_1)
            s_1 = Variable(torch.from_numpy(s_1)).type(torch.FloatTensor)
            epsilon = 0.4 + j * 0.01
            actions_value = target_net_1.forward(s_1).detach().numpy()
            if np.random.uniform() < epsilon: 
                a = np.argmax(actions_value)
            else:
                a = random.choice(list(data.unempty_cluster)) 
        
            choose_data_list = data.choose_data(a) # a list of data_id to be choosed

            s_2 = []
            action2_list = []
            print('cluster:', a)
            # print(data.unlabeled_num_of_each_cluster)

            
            for ind in range(len(choose_data_list)):
                state_2 = state.agent2_feature(choose_data_list[ind], actions_value[a])
                uncertain_feature_list.append(state_2[8])
                action_value_lsit.append(state_2[9])
                dist_sum_list.append(sum(state_2[10:]))

                s_2.append(state_2)
                choose_or_not = agent_2.choose_action(state_2, j) # 0 or 1
                data.update(choose_data_list[ind], choose_or_not)
                action2_list.append(choose_or_not)

            # print('s_2', s_2)
            # os.system('pause')
            s_2 = np.array(s_2)
            s_2 = Variable(torch.from_numpy(s_2)).type(torch.FloatTensor)
            print(action2_list)
            r_1, long_reward = env.feedback(a)
            s1_next = state.update()
            s_1 = s1_next 
            
            for ind in range(len(s_2)):
                r_2 = env.feedback2(choose_data_list[ind], action2_list[ind])
                r_2 += long_reward
                s2_next = state.agent2_feature(choose_data_list[ind], actions_value[a])
                agent_2.store_transition(s_2[ind], action2_list[ind], r_2, s2_next)
            
            if agent_2.memory_counter > MEMORY_CAPACITY_2:
                agent_2.Learn()

            
            # print('rest_cluster:', data.unlabeled_num_of_each_cluster)
            print('rest_data:', model.get_rest_unlabeled_data_effect())
        
        print('choosed_data_num:', data.labeled_num)

        # model.net.reset()
        # torch.save(agent_1.target_net, '\model_cluster_double_agent1_dist_20.pkl')
        torch.save(agent_2.target_net, '\model\model_cluster_double_agent2_dist_pretrain_20_v2.pkl')
    
    # x = range(len(uncertain_feature_list))
    # plt.plot(x, uncertain_feature_list)
    # plt.show()

    # y = range(len(action_value_lsit))
    # plt.plot(y, action_value_lsit)
    # plt.show()

    # z = range(len(dist_sum_list))
    # plt.plot(z, dist_sum_list)
    # plt.show()
    