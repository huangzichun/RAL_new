# k-means

import numpy as np
import matplotlib
import os
matplotlib.use('Agg')
from scipy.cluster.vq import *
import pylab
import data_input
import embedding
import json
from sklearn.decomposition import PCA

pylab.close()

FILENAME = 'A_train_newest.json'
TESTFILENAME = 'A_test_newest.json'
data_input = data_input.data_input(FILENAME, TESTFILENAME)
embed = embedding.embed(data_input)

with open(FILENAME) as f:
    word_pair_list = json.load(f)

all_embedding_sample = []
for word_pair in word_pair_list:
    all_embedding_sample.append(list(map(lambda x: x[0]-x[1], zip(embed.word2embed(word_pair[0]), embed.word2embed(word_pair[1])))))
all_embedding_sample = np.array(all_embedding_sample)
print(np.shape(all_embedding_sample)) 

color = ['g','r','c','m','y','k','w']
cluster_num = 3
res, idx = kmeans2(np.array(all_embedding_sample), cluster_num)


# print(np.shape(idx))
print("local centre points:\n",res)
all_embedding_sample = np.concatenate((all_embedding_sample, res),axis=0)

pca = PCA(n_components=10)
pca.fit(all_embedding_sample)
print(pca.explained_variance_ratio_)
all_embedding_sample_new = pca.transform(all_embedding_sample)

num = np.zeros(cluster_num)
plot_data = [[],[],[],[],[],[],[],[]]
for i in range(len(idx)):
    num[idx[i]] += 1
    plot_data[idx[i]].append(all_embedding_sample_new[i].tolist())

print(num)

# mark centroids as (X)
for i in range(len(res)):
    plot_data_i = np.array(plot_data[i])
    pylab.scatter(plot_data_i[:,0], plot_data_i[:,1], c=color[i])
    pylab.scatter(all_embedding_sample_new[i+len(idx),0], all_embedding_sample_new[i+len(idx),1], marker='x', s = 200, linewidths=3, c=color[i])
 
pylab.savefig('pic_3.png')


