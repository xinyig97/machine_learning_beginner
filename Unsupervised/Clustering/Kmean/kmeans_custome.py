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
				pass
				#self.centeroids[classification] = np.average(self.classifications[classification],axis=0)

			optimized = True



	def predict(self,data):
		pass
