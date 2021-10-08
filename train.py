# 模型训练部分

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import random
import os
import json



def train(model, data, agent, al, epoch, env, budget, MEMORY_CAPACITY, limited_label=100, initial_label=10):
    if limited_label < initial_label:
        print("error.")
        return
    rest_data_acc = []
    # single agent
    center_point = []
    for i_episode in range(epoch):
        print('========================================= epoch:', i_episode + 1)
        data.reset()
        data.first_random(initial_label)
        #model.net.reset()
        model.train()
        model.give_label()   # 这一步有问题，data.labeled_positive_negative_sample_of_each_cluster 里应该要人工的groundtruth和模型的预测混合的，取决于由谁标注
        acc_change = model.acc_change()
        print(acc_change)
        # if acc_change < -0.3:
        #     print('break!')
        #     os.system('pause')
        s = al.update()
        labeled_num = 0
        # begin to play the game

        while 1:
            s = Variable(torch.from_numpy(s)).type(torch.FloatTensor)
            a = agent.choose_action(s, -1)
    
            s_next, r = env.feedback(a) 
            agent.store_transition(s, a, r, s_next) 
            s = s_next 
            
            if agent.memory_counter > MEMORY_CAPACITY:
                print("========== agent learning ==========")
                agent.Learn()
            
            # print('rest_cluster:', data.unlabeled_num_of_each_cluster)
            # center_point.append(data.center_points.tolist())
            res_acc = model.get_rest_unlabeled_data_effect()
            print('rest_data:', res_acc)
            labeled_num += model.get_batch_size()
            rest_data_acc.append([res_acc, initial_label + labeled_num, a])


            if limited_label <= initial_label + labeled_num:
                print("reached the threshold... stop the game now")
                break

        # model.net.reset()
        torch.save(agent.target_net, 'model/model_cluster_20_v12.pkl')
    return rest_data_acc, env.add_reward, env.reward
    # with open("center.json","w") as f:
    #     json.dump(center_point, f)
