# 管理数据(标签，类别)

import json
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
from scipy.cluster.vq import *


class data(object):
    def __init__(self, FILENAME, TESTFILENAME, EMBEDFILE, EMBED_DIM, CLUSTER_NUM):
        # 数据形式:[word1 - word2, label, labeled or not, cluster] word以embedding形式构建
        self.batch_size = 16

        with open(FILENAME) as f:
            self.train_data = json.load(f)
        with open(TESTFILENAME) as f:
            self.test_data = json.load(f)

        np.random.shuffle(self.train_data)
        np.random.shuffle(self.test_data)

        with open(EMBEDFILE,'r') as f:
            self.embed = json.load(f)
        self.embed_dim = EMBED_DIM

        self.num_train_data = len(self.train_data)
        self.num_test_data = len(self.test_data)
        
        self.train_data_ = self.convert(self.train_data, self.num_train_data)
        self.test_data_ = self.convert(self.test_data, self.num_test_data)

        self.cluster_num = CLUSTER_NUM
        self.cluster_size = np.zeros(self.cluster_num)
        self.train_data_clustered, self.center_points = self.cluster(self.train_data_, self.cluster_num)
        self.unlabeled_num_of_each_cluster = np.zeros(self.cluster_num)
        self.labeled_num = 0
        self.cluster_list = range(self.cluster_num)
        self.unempty_cluster = set(self.cluster_list)
        self.labeled_positive_negative_sample_of_each_cluster = np.zeros((self.cluster_num, 2))
    
    def convert(self, data_list, num):  
        # 将数据化成[word1 - word2, true label, labeled or not and machine label, cluster]的形式
        # test_data的后两项无视
        new_data_list = np.zeros((num, self.embed_dim + 3))
        for i in range(num):
            embed1 = self.embed[data_list[i][0]]
            embed2 = self.embed[data_list[i][1]]
            new_data_list[i][:self.embed_dim] = list(map(lambda x: x[0] - x[1], zip(embed1, embed2)))
            new_data_list[i][self.embed_dim] = data_list[i][2]
            new_data_list[i][self.embed_dim + 1] = -1
            
        return new_data_list
    
    def cluster(self, data_list, cluster_num):
        cluster_data = data_list[:, :self.embed_dim]
        center_points, idx = kmeans2(cluster_data, cluster_num)
        data_list[:,self.embed_dim + 2] = idx
        for i in idx:
            self.cluster_size[i] += 1

        return data_list, center_points

    def train_loader(self, ): 
        data_list = []
        target_list = []
        for train_data in self.train_data_clustered:
            if train_data[self.embed_dim + 1] != -1:
                data_list.append(train_data[:self.embed_dim])
                target_list.append(train_data[self.embed_dim])

        data_list = np.array(data_list)
        data_list = Variable(torch.from_numpy(data_list)).type(torch.FloatTensor)
        target_list = np.array(target_list)
        target_list = Variable(torch.from_numpy(target_list)).type(torch.LongTensor)
        train_data = TensorDataset(data_list, target_list)
        print("training data size for model = {}".format(len(train_data)))
        train_loader = DataLoader(dataset = train_data, batch_size = self.batch_size, shuffle = True)
        return train_loader 
    
    def value_filter(self, action_values):
        min_action_values = min(action_values)
        for i in range(self.cluster_num):
            if not(i in self.unempty_cluster):
                action_values[i] = min_action_values - 1
        # print(action_values)

        return action_values
        
    def update(self, action): 
        # action is a number of a cluster, this function need to choose some data in that cluster randomly
        cluster_number = action
        choosed_num = 0
        choosed_data = []
        choosed_word_pair = []
        cluster_num = 0
        ind = 0
        for train_data in self.train_data_clustered:
            if int(train_data[self.embed_dim + 2]) == cluster_number and (int(train_data[self.embed_dim + 1]) == -1):
                cluster_num += 1

        for train_data in self.train_data_clustered:
            if (train_data[self.embed_dim + 2] == cluster_number) and (train_data[self.embed_dim + 1] == -1):
                choosed_num += 1
                train_data[self.embed_dim + 1] = 2
                choosed_data.append(train_data[:self.embed_dim + 1])
                word_pair = self.train_data[ind]
                choosed_word_pair.append(word_pair) 
            if choosed_num == self.batch_size:
                break
            ind += 1

        self.labeled_num += self.batch_size
        # print('cluster_number', cluster_number, 'cluster_num', cluster_num)
        # print('choosed_num', choosed_num)
       
        self.unlabeled_num_of_each_cluster[cluster_number] += -self.batch_size
        if choosed_num < self.batch_size or self.unlabeled_num_of_each_cluster[cluster_number] < self.batch_size:
            self.unempty_cluster.remove(cluster_number)
            print(cluster_number, 'removed')
            
        print('remain data:', self.unempty_cluster)
        # print('unlabeled_num:', self.unlabeled_num_of_each_cluster)    
        center_points = np.zeros((self.cluster_num, self.embed_dim))
        for train_data in self.train_data_clustered:
            if train_data[self.embed_dim + 1] == -1:
                center_points[int(train_data[self.embed_dim + 2]), :] += train_data[:self.embed_dim]
        
        for ind in range(len(center_points)):
            if self.unlabeled_num_of_each_cluster[ind] == 0:
                center_points[ind] = np.zeros(self.embed_dim)
            else:
                center_points[ind] = center_points[ind] / self.unlabeled_num_of_each_cluster[ind]
        
        self.center_points = center_points

        return choosed_data, choosed_word_pair

    def get_label_distribution(self):
        label_num = np.array(self.labeled_positive_negative_sample_of_each_cluster).sum(axis=1)
        unlabel_num = self.unlabeled_num_of_each_cluster
        res = np.array([x/y if y != 0 else 0.0 for x, y in zip(label_num, unlabel_num)])
        return res / res.sum()

    def get_unlabeled_data(self, ):
        unlabeled_data = []
        for train_data in self.train_data_clustered:
            if train_data[self.embed_dim + 1] == -1:
                unlabeled_data.append(train_data[:self.embed_dim + 1])

        unlabeled_data = np.array(unlabeled_data)
        return unlabeled_data
    
    def first_random(self, num):
        self.train_data_clustered[:num,self.embed_dim + 1] = 2
        for train_data in self.train_data_clustered[num:, :]:
            self.unlabeled_num_of_each_cluster[int(train_data[self.embed_dim + 2])] += 1 
        
        self.labeled_num += num
        for i in range(len(self.unlabeled_num_of_each_cluster)):
            if self.unlabeled_num_of_each_cluster[i] < self.batch_size:
                self.unempty_cluster.remove(i)
                print(i, 'removed')

        return 
    
    def reset(self, ): 
        np.random.shuffle(self.train_data_clustered)
        self.train_data_clustered[:, self.embed_dim + 1] = -1
        self.unlabeled_num_of_each_cluster = np.zeros(self.cluster_num)
        self.labeled_positive_negative_sample_of_each_cluster = np.zeros((self.cluster_num, 2))
        self.labeled_num = 0        
        self.unempty_cluster = set(self.cluster_list)
        return 