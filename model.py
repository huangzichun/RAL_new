# 机器分类器

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

# 分类器网络，输入word embedding（不训练），输出分类结果
class Classify_Net(nn.Module):
    def __init__(self, EMBEDDING_DIM):
        super(Classify_Net, self).__init__()
        # self.EMBEDDING_DIM = EMBEDDING_DIM
        # self.embedding = nn.Embedding(n_dict, EMBEDDING_DIM)
        self.fc1 = nn.Linear(EMBEDDING_DIM, 12)
        self.fc1.weight.data.normal_(0, 0.1)  
        self.out = nn.Linear(12, 2)
        self.out.weight.data.normal_(0, 0.1)  

    def forward(self, x):
        # emb = self.embedding(x)
        # emb = emb.view(-1, 2 * self.EMBEDDING_DIM)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        x = F.softmax(x)
        return x
    
    def uncertain(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        return x
    
    def reset(self, ):
        self.fc1.weight.data.normal_(0, 0.1)  
        self.out.weight.data.normal_(0, 0.1)  

class model(object):
    def __init__(self, data, EMBEDDING_DIM, device):
        self.net = Classify_Net(EMBEDDING_DIM).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = 0.01)
        self.loss_func = nn.CrossEntropyLoss() 
        self.epoch = 100
        self.device = device 
        self.data = data
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.acc = 0 #0.8

    def get_batch_size(self):
        return self.data.batch_size
       
    def train(self, ): # 使用所有有标签数据训练
        # print('start train!')
        loader = self.data.train_loader()

        for i in range(self.epoch):
            total_loss = 0 
            batch_num = 0
            for batch_id, batch_data in enumerate(loader):
                batch_num += 1
                batch_x, batch_y = batch_data
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_out = self.net.forward(batch_x)
                
                self.optimizer.zero_grad()
                loss = self.loss_func(batch_out, batch_y)
                total_loss += float(loss)
                loss.backward()
                self.optimizer.step()

            # if i % 24 == 0:
            #    print(i + 1, "epoch  loss:", total_loss / self.data.labeled_num)

            # if total_loss / self.data.labeled_num < 0.001:
            #     break
    
    def test(self, ): # 测试模型准确率上升
        acc_num = 0
        test_data_list = self.data.test_data_
        for test_data in test_data_list:
            x = test_data[:self.EMBEDDING_DIM]
            x = Variable(torch.from_numpy(x)).type(torch.FloatTensor).to(self.device)
            y = test_data[self.EMBEDDING_DIM]
            y_predict = self.net.forward(x).detach()
            
            if (y_predict[0] - y_predict[1]) * (y - 0.5) < 0:
                acc_num += 1
            
        return acc_num / len(test_data_list)
    
    def get_rest_unlabeled_data_effect(self, ):
        acc_num = 0
        unlabeled_data_list = self.data.get_unlabeled_data()
        for unlabeled_data in unlabeled_data_list:
            x = unlabeled_data[:self.EMBEDDING_DIM]
            x = Variable(torch.from_numpy(x)).type(torch.FloatTensor).to(self.device)
            y = unlabeled_data[self.EMBEDDING_DIM]
            y_predict = self.net.forward(x).detach()

            if (y_predict[0] - y_predict[1]) * (y - 0.5) < 0:
                acc_num += 1
        print("rest data effect = ")
        print(acc_num, len(unlabeled_data_list))
        return acc_num / len(unlabeled_data_list)

    def acc_change(self, ):
        new_acc = self.test()
        acc_change = new_acc - self.acc
        self.acc = new_acc
        if acc_change < 0:
            print()
        return acc_change

    def give_label(self, ground_truth=True):

        if not ground_truth:
            for train_data in self.data.train_data_clustered:
                x = train_data[:self.EMBEDDING_DIM]
                x = Variable(torch.from_numpy(x)).type(torch.FloatTensor).to(self.device)
                y = self.net.forward(x).detach().cpu().numpy()
                if train_data[self.EMBEDDING_DIM + 1] >= 0:
                    if y[0] > y[1]:
                        train_data[self.EMBEDDING_DIM + 1] = 0
                        self.data.labeled_positive_negative_sample_of_each_cluster[int(train_data[self.EMBEDDING_DIM + 2]), 0] += 1
                    else:
                        train_data[self.EMBEDDING_DIM + 1] = 1
                        self.data.labeled_positive_negative_sample_of_each_cluster[int(train_data[self.EMBEDDING_DIM + 2]), 1] += 1
        else:
            label_cid = self.data.train_data_clustered[:, self.EMBEDDING_DIM:]
            label_cid = label_cid[label_cid[:, 1] >= 0]
            for i in range(len(self.data.labeled_positive_negative_sample_of_each_cluster)):
                positive_label = label_cid[label_cid[:, 2] == i, 0].sum()
                self.data.labeled_positive_negative_sample_of_each_cluster[i, 1] = positive_label
                self.data.labeled_positive_negative_sample_of_each_cluster[i, 0] = len(label_cid[label_cid[:, 2] == i]) - positive_label
