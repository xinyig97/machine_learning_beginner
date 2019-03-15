import numpy as np 
from sklearn import preprocessing,neighbors
from sklearn.model_selection import cross_validate,train_test_split
import pandas as pd 
from math import sqrt
import warnings 
from collections import Counter 
import random

def k_nearest_neighbor(data,predict,k=3):
	print(len(data))
	if len(data) >= k:
		warnings.warn('k is set to a value less than total voting groups. ')
	distances = []
	for group in data:
		for features in data[group]:
			euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
			distances.append([euclidean_distance,group])

	votes = [i[1] for i in sorted(distances)[:k]]
	vote_result = Counter(votes).most_common(1)[0][0]
	confid = Counter(votes).most_common(1)[0][1]/k
	print (confid)
	return vote_result,confid

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)
test_size = 0.2
train_set = {2:[],4:[]}
test_set = {2:[],4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]
for i in train_data:
	train_set[i[-1]].append(i[:-1])
for i in test_data:
	test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set: # group is the key 
	print(group)
	for data in test_set[group]: # test_set[group] is the values associated with the key group
		vote = k_nearest_neighbor(train_set,data,k=5)
		if group == vote:
			correct += 1 
		total += 1

print(correct/total)

