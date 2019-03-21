import matplotlib.pyplot as plt 
from matplotlib import style
style.use('ggplot')
import numpy as np 
from sklearn.datasets.samples_generator import make_blobs
import random

# this is not stable omg 
centers = random.randrange(2,8)

X, y = make_blobs(n_samples= 50,centers = centers, n_features= 2)

# plt.scatter(X[:,0],X[:,1],s=150,linewidths = 5)
# plt.show()

colors = 10*['g','r','c','b','k']

# take every dataset as a centroid 
# take the average of data sets within the radius as the new centroid 
# do the same thing on the new centroid utill there's not enough change to the centroid
class Mean_Shift:
    def __init__(self,bandwidth = None, radius_norm_step = 100): # bandwidth is the key here 
        self.bandwidth = bandwidth
        self.radius_norm_step = radius_norm_step
    
    def fit(self,data):

        if self.bandwidth == None:
            all_data_centroid = np.average(data,axis = 0)
            all_data_norm = np.linalg.norm(all_data_centroid)
            self.bandwidth = all_data_norm / self.radius_norm_step

        centroids = {}

        for i in range(len(data)):
            centroids[i] = data[i]
        
        weights = [i for i in range(self.radius_norm_step)][::-1] # reverse the list 
        
        while True:
            new_centeroids = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                for featureset in data:
                    distance = np.linalg.norm(featureset - centroid)
                    if distance == 0:
                        distance = 0.000000001 
                    weight_index = int(distance/self.bandwidth)
                    if weight_index > self.radius_norm_step-1 :
                        weight_index = self.radius_norm_step -1 
                    # why? 
                    to_add = (weights[weight_index]**2) * [featureset]
                    in_bandwidth += to_add

                new_centroid = np.average(in_bandwidth,axis = 0)
                new_centeroids.append(tuple(new_centroid))
            uniques = sorted(list(set(new_centeroids)))
            to_pop = []

            # to get rid of very close points from the huge array it gets 
            for i in uniques:
                for ii in uniques:
                    if i == ii:
                        pass 
                    elif np.linalg.norm(np.array(i) - np.array(ii)) <= self.bandwidth:
                        to_pop.append(ii)
                        break # try avoiding adding mutilple nodes 
            for i in to_pop:
                try:
                    uniques.remove(i)
                except:
                    pass

            prev_centroids = dict(centroids)
            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])
            optimized = True 
            for i in centroids:
                if not np.array_equal(prev_centroids[i],centroids[i]):
                    optimized = False
                if not optimized:
                    break

            if optimized:
                break         

        self.centroids= centroids 

        self.classification = {}

        for i in range(len(self.centroids)):
            self.classification[i] = []
        
        for featureset in data:
            distance = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
            clasification = distance.index(min(distance))
            print(clasification)
            self.classification[clasification].append(featureset)

    def predict(self,data):
        distance = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        clasification = distance.index(min(distance))
        return clasification

clf = Mean_Shift()
clf.fit(X)
centroids = clf.centroids
plt.scatter(X[:,0],X[:,1],s=150)

for classification in clf.classification:
    color = colors[classification]
    for featureset in clf.classification[classification]:
        plt.scatter(featureset[0],featureset[1],marker='x',color = color, s=150,linewidths= 5)

for c in centroids:
    plt.scatter(centroids[c][0],centroids[c][1],color = 'k',marker='*',s=150)
plt.show()

            
        
