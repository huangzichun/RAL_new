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
        self.gamma_change = 1
        self.gamma_wrong_number = 1
        self.gamma_action = 0
        self.gamma_long = 1
        self.gamma_final = 1
        self.gamma_entropy = 10

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
        action_reward = change_reward = wrong_number_reward = long_reward = final_reward = 0
        print('action:', action)
        if action == self.old_action:
            action_reward = -1
        self.old_action = action
        choosed, choosed_word_pair = self.data.update(action)
        choosed = np.array(choosed)
        choosed_data = choosed[:, :self.embed_dim]
        choosed_target = choosed[:, self.embed_dim]
        choosed_data = np.array(choosed_data)
        choosed_data = Variable(torch.from_numpy(choosed_data)).type(torch.FloatTensor).to(self.device)
        old_label_of_choosed_data = self.Model.net(choosed_data).detach().cpu().numpy()
        old_label_of_choosed_data = self.convert(old_label_of_choosed_data)
        wrong_number_reward = self.compare_func(old_label_of_choosed_data, choosed_target)
        # add by hc， 当前模型错误率，如果错误率低，说明没必要学习，选择错误
        wrong_number_reward = wrong_number_reward/len(choosed_target)

        # cluster-level entropy
        old_label_distribution = self.get_label_distribution()
        print("reward training model")
        self.Model.train()
        new_label_of_choosed_data = self.Model.net(choosed_data).detach().cpu().numpy()
        new_label_of_choosed_data = self.convert(new_label_of_choosed_data)
        self.Model.give_label()
        new_label_distribution = self.get_label_distribution()
        entropy_reward = sum([x * math.log2(x/y) if y != 0 and x != 0 else 0 for x,y in zip(old_label_distribution, new_label_distribution)])

        old_label_of_choosed_data = self.Model.net(choosed_data).detach().cpu().numpy()
        old_label_of_choosed_data = self.convert(old_label_of_choosed_data)
        wrong_number_reward2 = self.compare_func(old_label_of_choosed_data, choosed_target)
        # add by hc， 当前模型错误率，如果错误率低，说明没必要学习，选择错误
        wrong_number_reward2 = wrong_number_reward2/len(choosed_target)
        change_reward = wrong_number_reward - wrong_number_reward2
        # change_reward = self.compare_func(old_label_of_choosed_data, new_label_of_choosed_data)

        long_reward = self.Model.acc_change()
        if self.Model.acc < 0.6:
            print('doubt data', choosed_word_pair)
            # os.system('pause')
        acc = self.Model.acc
        print('acc:', acc, 'on the validation data set (test data), currently label data =  ', self.data.labeled_num)

        if self.counter % self.budget == 0:
            final_acc = self.Model.get_rest_unlabeled_data_effect()
            final_reward = final_acc  # - 0.8
            print('final_acc:', final_acc)
            
            self.total_return_list.append(self.total_return)
            self.total_return = 0
        
        r = self.gamma_action * action_reward + self.gamma_change * change_reward + self.gamma_wrong_number * wrong_number_reward + self.gamma_long * long_reward + self.gamma_final * final_reward + self.gamma_entropy * entropy_reward
        print('reward:', r, 'action_reward:', self.gamma_action * action_reward, 'change_reward:', self.gamma_change * change_reward, 'wrong_number_reward:', self.gamma_wrong_number * wrong_number_reward, 'long_reward', self.gamma_long * long_reward, 'final_reward',  self.gamma_final * final_reward, 'entropy', self.gamma_entropy * entropy_reward)
        # print('long_reward:', self.gamma_long * long_reward, 'final_reward:', self.gamma_final * final_reward)
        self.reward.append(r)
        self.add_reward.append([action_reward, change_reward, wrong_number_reward, long_reward, final_reward, entropy_reward])
        self.total_return += r   

        self.counter += 1

        self.state = self.al.update()
        s_next = self.state 
        
        return s_next, r
    
    def convert(self, label):
        new_label = np.zeros(len(label))
        for i in range(len(label)):
            label_ = label[i]
            if label_[0] < label_[1]:
                new_label[i] = 1
        
        return new_label

    def get_label_distribution(self):
        return self.data.get_label_distribution()
    
    def compare_func(self, label1, label2):
        num = 0
        for i in range(len(label1)):
            if label1[i] != label2[i]:
                num += 1
        
        return num