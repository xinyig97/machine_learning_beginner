# own knn 
from math import sqrt
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import style 
import warnings 
from collections import Counter 

style.use('fivethirtyeight')

dataset = {'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]


def k_nearest_neighbor(data,predict,k=3):
	print(len(data))
	if len(data) >= k:
		warnings.warn('k is set to a value less than total voting groups. ')
	#skeptical
	distances = []
	for group in data:
		for features in data[group]:
			# euclidean_distance = np.sqrt(np.sum((np.array(features)-np.array(predict))**2))
			#faster than the one we wrote before by using np.sqrt ,np.sum, and np.array 
			# a even fatser one :
			euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
			distances.append([euclidean_distance,group])

	votes = [i[1] for i in sorted(distances)[:k]]
	# print(Counter(votes).most_common(1))
	# print(Counter(votes).most_common(1)[0])
	# print(Counter(votes).most_common(1)[0][0])
	vote_result = Counter(votes).most_common(1)[0][0]
	confidence = Counter(votes).most_common(1)[0][1]/k
	return vote_result,confidence


result,confidence = k_nearest_neighbor(dataset,new_features,k=3)
print(result,confidence)

for i in dataset:
	for ii in dataset[i]:
		plt.scatter(ii[0],ii[1],s =100, color=i)
plt.scatter(new_features[0],new_features[1],color=result)
plt.show()

