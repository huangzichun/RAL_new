# state feature

import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from sklearn.neighbors import KDTree

class state(object):
    def __init__(self, data, Model, EMBEDDING_DIM, device):
        self.Model = Model
        self.data = data
        self.cluster_num = self.data.cluster_num
        self.dim = EMBEDDING_DIM
        self.device = device

        self.state_feature = []
        self.kdTree = KDTree(self.data.train_data_clustered[:, :self.dim], leaf_size=40)
        # self.data_matrix = self.data.center_points
        # label_dict = {}
        # self.dis_env = DisEnv.DisEnv(self.embed_matrix, leaf_size=400, k=3, label_dict=label_dict)
        # self.dis_env.init_env()

    def uncertainty(self, ):  # 模型输出，衡量不确信度
        # cluster_centers = self.data.center_points
        # cluster_centers = Variable(torch.from_numpy(cluster_centers)).type(torch.FloatTensor).to(self.device)
        # output = self.Model.net.uncertain(cluster_centers).detach()
        # uncertainty = abs(output[:, 0] - output[:, 1])
        # norm = max(uncertainty)
        # return(uncertainty / norm)

        x = self.data.train_data_clustered[:, :self.dim]
        x = Variable(torch.from_numpy(x)).type(torch.FloatTensor).to(self.device)
        y = self.Model.net.uncertain(x).detach()
        uncertainty = np.zeros(self.cluster_num)
        i = 0

        for ind in self.data.train_data_clustered:
            if ind[self.dim + 1] == -1:
                uncertainty[int(ind[self.dim + 2])] += abs(y[i, 0] - y[i, 1])
            i += 1
        
        
        for i in range(self.cluster_num):
            if self.data.unlabeled_num_of_each_cluster[i] == 0:
                uncertainty[i] = 0
            else:
                uncertainty[i] = uncertainty[i] / self.data.unlabeled_num_of_each_cluster[i]

        #
        norm = max(uncertainty) 
        uncertainty = uncertainty / norm     
        
        return uncertainty


    def cluster_entropy(self, ):
        entropy_data = self.data.labeled_positive_negative_sample_of_each_cluster
        entropy = np.zeros(self.cluster_num)
        ind = 0
        for each_cluster_entropy in entropy_data:
            negative_num = each_cluster_entropy[0]
            positive_num = each_cluster_entropy[1]
            total_num = negative_num + positive_num
            if total_num == 0:
                entropy[ind] = 1
            
            else:
                negative_ratio = negative_num / total_num
                positive_ratio = positive_num / total_num

                if negative_ratio == 0 or positive_ratio == 0:
                    entropy[ind] = 0
                else:
                    entropy[ind] = - negative_ratio * math.log2(negative_ratio) - positive_ratio * math.log2(positive_ratio)
            ind += 1
        
        return entropy
        
    def update(self, ): 
        state_uncertainty = self.uncertainty().tolist()
        state_cluster_entropy = self.cluster_entropy().tolist()
        self.state_feature = []
        self.state_feature.extend(state_uncertainty)
        self.state_feature.extend(state_cluster_entropy)

        # self.state_feature[: self.cluster_num * self.dim] = self.data.center_points.reshape(self.cluster_num * self.dim)
        # self.state_feature[self.cluster_num * self.dim: self.cluster_num * self.dim + self.cluster_num] = state_uncertainty
        # self.state_feature[self.cluster_num * self.dim + self.cluster_num:self.cluster_num * self.dim + self.cluster_num * 2] = state_cluster_entropy
        # self.state_feature[-self.cluster_num: ] = 
        # print('state:', self.state_feature)

        state_feature = np.array(self.state_feature)
        # print('feature1:', state_feature)
        return state_feature
    
    def distance(self, data_point, k):
        dist, ind = self.kdTree.query(data_point.reshape(1,-1), k=k)
        return dist