# agent2, 提供是否选择当前数据的策略

from agent import Q_Net
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.autograd import Variable


class agent2(object):
    def __init__(self, Data, N_STATES, N_ACTIONS, device):
        self.eval_net, self.target_net = Q_Net(N_STATES, N_ACTIONS), Q_Net(N_STATES, N_ACTIONS)

        self.TARGET_REPLACE_ITER = 10
        self.BATCH_SIZE = 32
        self.LR = 0.01
        self.MEMORY_CAPACITY = 320
        self.GAMMA = 0.9

        self.learn_step_counter = 0 
        self.memory_counter = 0 
        self.memory = np.zeros((self.MEMORY_CAPACITY,N_STATES * 2 + 2)) 

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr = self.LR)
        self.loss_func = nn.MSELoss() 

        self.N_STATES = N_STATES
        self.Data = Data
        self.device = device

    def choose_action(self, x, j):
        # x = x.squeeze(-1)
        x = Variable(torch.from_numpy(x)).type(torch.FloatTensor)
        x = x.to(self.device)
        epsilon = 0.4 + j * 0.01
        if np.random.uniform() < epsilon: 
            actions_value = self.eval_net.forward(x).detach().numpy()
            action = np.argmax(actions_value)
            print(actions_value)
        else:
            action = random.choice([0, 1])

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
        b_memory = self.memory[sample_index, :]

        b_s = torch.FloatTensor(b_memory[:, :self.N_STATES]).to(self.device)
        b_a = torch.LongTensor(b_memory[:, self.N_STATES:self.N_STATES+1].astype(int)).to(self.device)
        b_r = torch.FloatTensor(b_memory[:, self.N_STATES+1:self.N_STATES+2]).to(self.device)
        b_s_next = torch.FloatTensor(b_memory[:, -self.N_STATES:]).to(self.device)

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_next).detach()
        q_target = b_r + self.GAMMA * q_next.max(1)[0].view(self.BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()