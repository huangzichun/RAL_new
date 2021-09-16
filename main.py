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

import env 
import agent 
import agent2
import data
import model
import state
import train
import test

import warnings
import sys

# sys.stdout = open('out.log', 'a', encoding='utf-8')

warnings.filterwarnings("ignore")

random.seed(114514)

epoch = 5
CLUSTER_NUM = 20
EMBEDDING_DIM = 25
budget = 50
MEMORY_CAPACITY = budget
MEMORY_CAPACITY_2 = 320
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

k = 5
N_STATES_2 = 10 + k
N_ACTIONS_2 = 2
agent2 = agent2.agent2(train_data, N_STATES_2, N_ACTIONS_2, device)

train.train(train_Model, train_data, Agent, agent2, train_state, epoch, Env, budget, MEMORY_CAPACITY, MEMORY_CAPACITY_2)
# Q_Net_loss = Agent.Q_Net_loss
# x = range(len(Q_Net_loss))
# plt.plot(x, Q_Net_loss)
# plt.show()

# total_return = Env.total_return_list
# y = range(len(total_return))
# plt.plot(y, total_return)
# plt.show()

FILENAME = 'B_train_newest.json'
TESTFILENAME = 'B_test_newest.json'

test_data = data.data(FILENAME, TESTFILENAME, EMBEDDINGFILE, EMBEDDING_DIM, CLUSTER_NUM)
test_Model = model.model(test_data, EMBEDDING_DIM, device)
test_state = state.state(test_data, test_Model, EMBEDDING_DIM, device)
test_Env = env.env(test_data, test_state, test_Model, EMBEDDING_DIM, budget, device)
test.test(test_data, test_Model, test_Env, test_state, budget)