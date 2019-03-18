import matplotlib.pyplot as plt 
from matplotlib import style 
style.use('ggplot')
import numpy as np 
from sklearn.cluster import KMeans 
from sklearn import preprocessing 
from sklearn.model_selection import cross_validate 
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
print(df.head())
