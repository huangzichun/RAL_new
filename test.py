# 模型训练部分

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import random

def test(data, model, env, state, budget):
    model.net.reset()
    data.reset()
    data.first_random(800)
    model.train()
    model.give_label()
    print(model.acc_change())
    target_net_1 = torch.load('\model\model_cluster_20_v7.pkl')
    taregt_net_2 = torch.load('\model\model_cluster_double_agent2_dist_pretrain_20_v2.pkl')
    s_1 = state.update() 
    s_1 = Variable(torch.from_numpy(s_1)).type(torch.FloatTensor)
    for j in range(budget):
        # j = data.labeled_num - 1000
        epsilon = 0.4 + j * 0.01
        actions_value = target_net_1.forward(s_1).detach().numpy()
        if np.random.uniform() < epsilon: 
            a = np.argmax(actions_value)
        else:
            a = random.choice(list(data.unempty_cluster)) 
       
        choose_data_list = data.choose_data(a) # a list of data_id to be choosed
        s_2 = []
        action2_list = []
        print(a)
        print(data.unlabeled_num_of_each_cluster)
        for ind in range(len(choose_data_list)):
            state_2 = state.agent2_feature(choose_data_list[ind], actions_value[a])
            s_2.append(state_2)
            state_2 = Variable(torch.from_numpy(state_2)).type(torch.FloatTensor)
            if np.random.uniform() < epsilon: 
                action_value = taregt_net_2.forward(state_2).detach().numpy()
                choose_or_not = np.argmax(action_value)
                print(action_value)
            else:
                choose_or_not = random.choice([0, 1])
            data.update(choose_data_list[ind], choose_or_not)
            action2_list.append(choose_or_not)

        s_2 = np.array(s_2)
        s_2 = Variable(torch.from_numpy(s_2)).type(torch.FloatTensor)
        print(action2_list)
        r_1 = env.feedback(a)
        s1_next = state.update()
        s_1 = s1_next 
        s_1 = Variable(torch.from_numpy(s_1)).type(torch.FloatTensor)
        actions_value = target_net_1.forward(s_1).detach().numpy()

        for ind in range(len(s_2)):
            r_2 = env.feedback2(choose_data_list[ind], action2_list[ind])
            s2_next = state.agent2_feature(choose_data_list[ind], actions_value[a])

    print(model.get_rest_unlabeled_data_effect())
    print('choosed_data_num:', data.labeled_num)
        
        