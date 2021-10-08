# 提供policy，目前采用的是DQN

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
import os

class Q_Net(nn.Module):
    # 定义一个Q网络的类，输入：当前状态；输出：每种action能获得的return
    def __init__(self, N_STATES, N_ACTIONS):
        super(Q_Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, int(1.5 * N_ACTIONS))
        self.fc1.weight.data.normal_(0, 0.1)  
        self.out = nn.Linear(int(1.5 * N_ACTIONS), N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self, data, N_STATES, N_ACTIONS, MEMORY_CAPACITY, device):
        # DQN有两个神经网络，一个是eval_net一个是target_net
        # 两个神经网络相同，参数不同，每隔一段时间把eval_net的参数转化成target_net的参数，产生延迟的效果
        self.eval_net, self.target_net = Q_Net(N_STATES, N_ACTIONS).to(device), Q_Net(N_STATES, N_ACTIONS).to(device)

        self.BATCH_SIZE = 16
        self.LR = 0.01
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.TARGET_REPLACE_ITER = 20
        self.GAMMA = 0.9
        self.device = device

        self.learn_step_counter = 0 
        self.memory_counter = 0 
        self.memory = np.zeros((self.MEMORY_CAPACITY, N_STATES * 2 + 2)) 

        # 记忆库初始化为全0，存储两个state的数值加上一个a(action)和一个r(reward)的数值
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr = self.LR)
        self.loss_func = nn.MSELoss() 
        # self.epsilon = 0.9

        self.N_STATES = N_STATES
        self.data = data
        self.Q_Net_loss = []
   
    def choose_action(self, x, j): 
        epsilon = 0.8
        x = x.squeeze(-1)
        x = x.to(self.device)
        if np.random.uniform() < epsilon: 
            # print('state:', x)
            # print(self.eval_net.forward(x))
            actions_value = self.eval_net.forward(x).detach().numpy()
            actions_value = self.data.value_filter(actions_value)  
            action = np.argmax(actions_value)
            # print('actions_value:', actions_value)
        else:
            action = random.choice(list(self.data.unempty_cluster))
            
        return action    

    def store_transition(self, s, a, r, s_next):
        s = s.squeeze(-1)
        # s_next = s_next.squeeze(-1)
        transition = np.hstack((s, [a, r], s_next))
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
    
    def Learn(self):
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)

        print("train agent with {} instances".format(sample_index))

        b_memory = self.memory[sample_index, :]
        
        b_s = torch.FloatTensor(b_memory[:, :self.N_STATES]).to(self.device)
        b_a = torch.LongTensor(b_memory[:, self.N_STATES:self.N_STATES+1].astype(int)).to(self.device)
        b_r = torch.FloatTensor(b_memory[:, self.N_STATES+1:self.N_STATES+2]).to(self.device)
        b_s_next = torch.FloatTensor(b_memory[:, self.N_STATES+2:]).to(self.device) 

        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_next).detach()  
        # print('state:', b_s)
        # print('q_eval:', q_eval)
        # print('q_next:', q_next)

        for i in range(self.BATCH_SIZE):
            q_next[i] = self.data.value_filter(q_next[i])

        q_target = b_r + self.GAMMA * q_next.max(1)[0]  
        q_target = q_target[0]
        loss = self.loss_func(q_eval, q_target)
        q_target_numpy = q_target.numpy()
        q_eval_numpy = q_eval.detach().numpy()
        print('loss:', loss)
        if self.learn_step_counter > 1:
            self.Q_Net_loss.append(math.sqrt(float(loss)))

        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward() 
        self.optimizer.step()
    