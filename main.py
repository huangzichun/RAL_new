import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import random
import json
import copy

import env 
import agent 
import data
import model
import state
import train
import test

import warnings
import sys

# sys.stdout = open('out.log', 'a', encoding='utf-8')

warnings.filterwarnings("ignore")

random.seed(37)

epoch = 500
CLUSTER_NUM = 20
EMBEDDING_DIM = 25
budget = 32  # 10
MEMORY_CAPACITY = budget
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

FILENAME = 'A_train_newest.json'
TESTFILENAME = 'A_test_newest.json'
EMBEDDINGFILE = 'embedding25.json'

train_data = data.data(FILENAME, TESTFILENAME, EMBEDDINGFILE, EMBEDDING_DIM, CLUSTER_NUM)
train_Model = model.model(train_data, EMBEDDING_DIM, device)
train_state = state.state(train_data, train_Model, EMBEDDING_DIM, device)
Env = env.env(train_data, train_state, train_Model, EMBEDDING_DIM, budget, device)

N_STATES = len(Env.state) 
N_ACTIONS = CLUSTER_NUM
Agent = agent.DQN(train_data, N_STATES, N_ACTIONS, MEMORY_CAPACITY, device)

res_info, reward_info, sum_reward_info = train.train(train_Model, train_data, Agent, train_state, epoch, Env, budget, MEMORY_CAPACITY)
Q_Net_loss = Agent.Q_Net_loss
x = range(len(Q_Net_loss))
plt.subplot(2, 3, 1)
plt.title('Q loss')
plt.plot(x, Q_Net_loss)
# plt.show()

total_return = Env.total_return_list
y = range(len(total_return))
plt.subplot(2, 3, 2)
plt.title('total return')
plt.plot(y, total_return)
# plt.show()

plt.subplot(2, 3, 3)
x = range(len(res_info))
y = np.array(res_info)[:, 0]
plt.title('rest acc')
plt.plot(x, np.array(res_info)[:, 0],  color='skyblue', label='rest acc')
# plt.plot(x, np.array(res_info)[:, 1] / 100, color='blue', label='label num')

plt.subplot(2, 3, 4)
x = range(len(res_info))
y = np.array(res_info)[:, 2]
plt.plot(x, y)
plt.title('actions')

plt.subplot(2, 3, 5)
x = range(len(sum_reward_info))
y = sum_reward_info
plt.plot(x, y)
plt.title('total reward')

plt.subplot(2, 3, 6)
plt.title('reward')
x = range(len(reward_info))
# plt.plot(x, np.array(reward_info)[:, 0],  color='skyblue', label='action')
plt.plot(x, np.array(reward_info)[:, 1], color='blue', label='change')
plt.plot(x, np.array(reward_info)[:, 2],  color='red', label='wrong')
plt.plot(x, np.array(reward_info)[:, 3], color='green', label='long')
plt.plot(x, np.array(reward_info)[:, 4],  color='black', label='final')
plt.plot(x, np.array(reward_info)[:, 5], color='yellow', label='entropy')

plt.legend()
plt.show()
# label=106 acc=0.8250878778748619 std= 0.013942691053742583

exit(0)
FILENAME = 'B_train_newest.json'
TESTFILENAME = 'B_test_newest.json'

test_data = data.data(FILENAME, TESTFILENAME, EMBEDDINGFILE, EMBEDDING_DIM, CLUSTER_NUM)
test_data.first_random(200)
test_data_copy = copy.copy(test_data)

test_Model_random = model.model(test_data, EMBEDDING_DIM, device)
test_state = state.state(test_data, test_Model_random, EMBEDDING_DIM, device)
test_Env = env.env(test_data, test_state, test_Model_random, EMBEDDING_DIM, budget, device)
epsilon = 0
test_data_acc_random, rest_data_acc_random = test.test(test_data, test_Model_random, test_Env, test_state, epsilon, budget)

test_Model_RAL = model.model(test_data_copy, EMBEDDING_DIM, device)
test_state = state.state(test_data_copy, test_Model_RAL, EMBEDDING_DIM, device)
test_Env = env.env(test_data_copy, test_state, test_Model_RAL, EMBEDDING_DIM, budget, device)
epsilon = 1
test_data_acc_RAL, rest_data_acc_RAL = test.test(test_data_copy, test_Model_RAL, test_Env, test_state, epsilon, budget)

print('random:', test_data_acc_random, rest_data_acc_random)
print('RAL:', test_data_acc_RAL, rest_data_acc_RAL)