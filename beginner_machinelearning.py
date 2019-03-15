# machine learning : not hard coding for machine to do something 
# regression: take in a bunch of data and trying to find the line that best fit the data 

import pandas as pd 
import quandl 
import math
import numpy as np 
from sklearn import preprocessing, svm 
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import LinearRegression 


df = quandl.get("WIKI/GOOGL")
# print(df.head())
# define the deatues

# those are features, not s 
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

df['HL_PCT'] = ((df['Adj. High']-df['Adj. Close'])/ df['Adj. Close']) * 100.0

df['PCT_Change'] = ((df['Adj. Close']-df['Adj. Open'])/ df['Adj. Open']) * 100.0

df = df[['Adj. Close','HL_PCT','PCT_Change','Adj. Volume']]
#print(df.head()) 
forecast_col = 'Adj. Close'
df.fillna(-99999,inplace = True) #sub nan with dummy data, since in ml, you dont want to lose data 

forecast_out = int(math.ceil(0.01*len(df)))  # number of days out 
print(forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['label'],1)) # data uses captial X 
y = np.array(df['label'])

X = preprocessing.scale(X) # scale all values 
y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)


# can use multiple treads and parallel programming!! n_jobs yea!!!
# classifier= LinearRegression()
# classifier.fit(X_train,y_train)
# accuracy = classifier.score(X_test,y_test)


classifier= svm.SVR(kernel = 'poly') # what is a kernel 
classifier.fit(X_train,y_train)
accuracy = classifier.score(X_test,y_test)

print(accuracy)





