# 模型训练部分

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import random

def test(data, model, env, al, epsilon, budget):
    model.net.reset()
    model.train()
    model.give_label()
    print(model.acc_change())
    target_net = torch.load('\model\model_cluster_20_v12.pkl')
    s = al.update()
    for j in range(budget):
        # 根据dqn来接受现在的状态，得到一个行为
        # 行为：挑选数据 
        # print(s)
        
        s = Variable(torch.from_numpy(s)).type(torch.FloatTensor)
        s = s.squeeze(-1)

        # epsilon = 0.8
        if np.random.uniform() < epsilon: 
            actions_value = target_net.forward(s).detach().numpy()
            actions_value = data.value_filter(actions_value)  
            action = np.argmax(actions_value)
        else:
            action = random.choice(list(data.unempty_cluster))

        # baseline3采用最大不确信度 
        # actions_value = al.state_feature[:data.cluster_num]
        # actions_value = data.value_filter(actions_value)
        # action = np.argmax(actions_value)

        s_next, r = env.feedback(action) 
        s = s_next 
    
    # print(model.get_rest_unlabeled_data_effect())
    return(model.acc, model.get_rest_unlabeled_data_effect())
        
        