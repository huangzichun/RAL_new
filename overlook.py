import json
import numpy as np
import data_input
import embedding

FILENAME = 'A_train_newest.json'
TESTFILENAME = 'A_test_newest.json'
data_input = data_input.data_input(FILENAME, TESTFILENAME)
embed = embedding.embed(data_input)

with open(FILENAME) as f:
    word_pair_list = json.load(f)

positive_sample = []
negative_sample = []
for word_pair in word_pair_list:
    if word_pair[2] == 1:
        positive_sample.append([embed.word2embed(word_pair[0]), embed.word2embed(word_pair[1])])
    else: 
        negative_sample.append([embed.word2embed(word_pair[0]), embed.word2embed(word_pair[1])])

positive_cosine = negative_cosine = 0
positive_vec = np.zeros(50)
negative_vec = np.zeros(50)
for word_pair in positive_sample:
    a = np.array(word_pair[0])
    b = np.array(word_pair[1])
    positive_cosine += np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    positive_vec += a - b

positive_cosine = positive_cosine / len(positive_sample)
positive_vec = positive_vec / len(positive_sample)

for word_pair in negative_sample:
    a = np.array(word_pair[0])
    b = np.array(word_pair[1])
    negative_cosine += np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    negative_vec += a - b

negative_cosine = negative_cosine / len(negative_sample)
negative_vec = negative_vec / len(negative_sample)

print('positive:', len(positive_sample), 'negative:', len(negative_sample))
print(positive_cosine, negative_cosine)
print(positive_vec)
print(negative_vec)
print(positive_vec + negative_vec)
