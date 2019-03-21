import matplotlib.pyplot as plt 
from matplotlib import style 
style.use('ggplot')
import numpy as np 
from sklearn.cluster import MeanShift
from sklearn import preprocessing 
import pandas as pd

'''
Pclass Passenger Class ( 1 =1st , 2 =2nd , 3= 3rd)
survival Survival (0 - no , 1 - yes )
name Name 
sex Sex 
age Age 
sibsp Number of siblings . spouses abroad 
parch Number of parents . children abroad 
ticket Ticket Number 
fare Passenger Fare 
cabin Cabin 
embarked Port of Emvarkation (C = cherbourg, Q = queenstown s = southampton )
boat Liftboat 
body Body indentification number 
home.dest Homes/Destinatopn
'''

df = pd.read_excel('titanic.xls')
original_df = pd.DataFrame.copy(df) # get a copy of the dataframe , for later reference 

df.drop(['body','name'],1,inplace=True)
#df.convert_objects(convert_numeric=True)
df.fillna(0,inplace=True)

# all machine leanrig need numerical data, need to convert non numerical data to numerical 
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
df = handle_non_numerical_data(df)
# print(df.head())

# df.drop(['sex'],1,inplace=True)

X = np.array(df.drop(['survived'],1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])
clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_


original_df['cluster_group'] = np.nan 

for i in range(len(X)):
    original_df['cluster_group'].iloc[i] = labels[i]  #iloc === indexing in the dataframe-> i'th row in the dataframe 

n_clusters_ = len(np.unique(labels))
survival_rates = {}

for i in range(n_clusters_):
    temp_df = original_df[(original_df['cluster_group'] == float(i))]
    survival_cluster = temp_df[(temp_df['survived']==1)]
    survival_rate = len(survival_cluster)/len(temp_df)
    survival_rates[i] = survival_rate
print(survival_rates)

# for i in survival_rates:
#     print('survivaed: ' + str(i))
#     print(original_df[(original_df['cluster_group']== i)])

