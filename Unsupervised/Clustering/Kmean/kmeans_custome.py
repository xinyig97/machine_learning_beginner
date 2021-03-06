import matplotlib.pyplot as plt 
from matplotlib import style
style.use('ggplot')
import numpy as np 

X = np.array([[1,2],
				[1.5,1.8],
				[5,8],
				[8,8],
				[1,0.6],
				[9,11]])

plt.scatter(X[:,0],X[:,1],s=150,linewidths = 5)
plt.show()

colors = 10*['g','r','c','b','k']

class k_means:
	def __init__(self,k=2, tol = 0.001, max_iter = 300): # tol is how much the centeroids is gonna change
		self.k = k
		self.tol = tol 
		self.max_iter = max_iter

	def fit(self,data):
		self.centeroids = {}
		for i in range(self.k):
			self.centeroids[i] = data[i]

		for i in range(self.max_iter):
			self.classifications = {}

			for j in range(self.k):
				self.classifications[j] = []

			for featureset in data:
				distances = [np.linalg.norm(featureset-self.centeroids[centeroids]) for centeroids in self.centeroids]
				classification = distances.index(min(distances))
				self.classifications[classification].append(featureset)

			prev_centroids = dict(self.centeroids)

			for classification in self.classifications:
				self.centeroids[classification] = np.average(self.classifications[classification],axis=0)

			optimized = True

			for c in self.centeroids:
				original_centeroid = prev_centroids[c]
				current_centeroid = self.centeroids[c]
				if np.sum((current_centeroid - original_centeroid)/original_centeroid * 100.0) > self.tol:
					optimized = False
			if optimized:
				break

	def predict(self,data):
		distances = [np.linalg.norm(data-self.centeroids[centeroids]) for centeroids in self.centeroids]
		classification = distances.index(min(distances))
		return classification

clf = k_means()
clf.fit(X)
for centroid in clf.centeroids:
	plt.scatter(clf.centeroids[centroid][0],clf.centeroids[centroid][1],marker = 'o', color = 'k', s =150, linewidths= 5)

for classification in clf.classifications:
	color = colors[classification]
	for featureset in clf.classifications[classification]:
		plt.scatter(featureset[0],featureset[1],marker='x',color = color, s=150,linewidths= 5)


unknowns = np.array([[1,3],[8,9],[5,4],[6,4],[0,3]])
for unknown in unknowns:
	classification = clf.predict(unknown)
	plt.scatter(unknown[0],unknown[1],marker='*',color = colors[classification],s=150,linewidths=5)


plt.show()