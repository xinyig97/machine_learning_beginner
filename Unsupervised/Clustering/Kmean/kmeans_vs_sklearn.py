import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
import pandas as pd 

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

def handle_non_numerical_data(df):
	columns = df.columns.values
	for column in columns:
		text_digit_vals = {}
		def convert_to_int(val):
			return text_digit_vals[val]
		if df[column].dtype != np.int64 and df[column].dtype != np.float64:
			column_contents = df[column].values.tolist()
			unique_elements = set(column_contents)
			x = 0
			for unique in unique_elements:
				if unique not in text_digit_vals:
					text_digit_vals[unique] = x 
					x += 1
			df[column] = list(map(convert_to_int,df[column]))
	return df 

df = pd.read_excel('titanic.xls')
df.drop(['body','name'],1,inplace=True)
df.fillna(0,inplace=True)
df = handle_non_numerical_data(df)
df.drop(['ticket','home.dest'],1,inplace=True)
X = np.array(df.drop(['survived'],1).astype(float))
print(df.head())
X = preprocessing.scale(X)
y = np.array(df['survived'])
clf = k_means()
clf.fit(X)
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1,len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction == y[i]:
        correct += 1 

print(correct/len(X))