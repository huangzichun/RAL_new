# 基于上下位词标注的环境

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math
import os

class env(object):
    def __init__(self, data, al, Model, embed_dim, budget, device):
        self.counter = 1
        self.gamma_wrong_number = 0.1
        self.gamma_action = 1
        self.gamma_long = 100 
        self.gamma_final = 10
        self.gamma_entropy = 1

        self.data = data
        self.al = al
        self.Model = Model 
        self.embed_dim = embed_dim
        self.budget = budget
        self.device = device
       
        self.state = self.al.update()
        self.reward = []
        self.add_reward = []
        self.total_return_list = []
        self.total_return = 0
        self.old_action = -1

    def feedback(self, action):  
        entropy_reward = action_reward = long_reward = final_reward = 0
        if action == self.old_action:
            action_reward = -1
        self.old_action = action

        entropy = self.al.cluster_entropy()
        entropy_reward = entropy[action]

        self.Model.train()
        self.Model.give_label()

        long_reward = self.Model.acc_change()
        if self.Model.acc < 0.6:
            print('doubt data', choosed_word_pair)
            os.system('pause')
        acc = self.Model.acc
        print('acc:', acc, 'label_num:', self.data.labeled_num)

        if self.counter % self.budget == 0:
            final_acc = self.Model.get_rest_unlabeled_data_effect()
            final_reward = final_acc - 0.8
            print('final_acc:', final_acc)
            
            # self.total_return_list.append(self.total_return)
            # self.total_return = 0
        
        r = self.gamma_action * action_reward  + self.gamma_entropy * entropy_reward + self.gamma_long * long_reward + self.gamma_final * final_reward 
        print('reward:', r, 'action_reward:', self.gamma_action * action_reward, 'entropy_reward:', self.gamma_entropy * entropy_reward)
        print('long_reward:', self.gamma_long * long_reward, 'final_reward:', self.gamma_final * final_reward)
        self.counter += 1

        return r, long_reward
    
    def feedback2(self, choose_data_id, choose_or_not):
        r = 0
        x = self.data.train_data_clustered[choose_data_id,  :self.embed_dim]
        x = Variable(torch.from_numpy(x)).type(torch.FloatTensor).to(self.device)
        y = self.data.train_data_clustered[choose_data_id, self.embed_dim]
        y_predict = self.Model.net(x).detach()
        if (y_predict[0] - y_predict[1]) * (y - 0.5) > 0: # wrong!
            if choose_or_not == 1:
                r += 1
        else:
            if choose_or_not == 0:
                r += 1
        
        # if choose_or_not == 1:
        #     r -= 0.1
        
        return r
    
    def convert(self, label):
        new_label = np.zeros(len(label))
        for i in range(len(label)):
            label_ = label[i]
            if label_[0] < label_[1]:
                new_label[i] = 1
        
        return new_label
    
    def compare_func(self, label1, label2):
        num = 0
        for i in range(len(label1)):
            if label1[i] != label2[i]:
                num += 1
        
        return num