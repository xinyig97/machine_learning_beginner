# machine learning : not hard coding for machine to do something 
# regression: take in a bunch of data and trying to find the line that best fit the data 

import pandas as pd 
import quandl 
import math
import datetime
import numpy as np 
from sklearn import preprocessing, svm 
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt 
from matplotlib import style
import pickle # save the classifier

style.use('ggplot')


df = quandl.get("WIKI/GOOGL")
# print(df.head())
# define the deatues

# those are features, not s 
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

df['HL_PCT'] = ((df['Adj. High']-df['Adj. Close'])/ df['Adj. Close']) * 100.0

df['PCT_Change'] = ((df['Adj. Close']-df['Adj. Open'])/ df['Adj. Open']) * 100.0

#          price         x        x           x
df = df[['Adj. Close','HL_PCT','PCT_Change','Adj. Volume']]
#print(df.head()) 
forecast_col = 'Adj. Close'
df.fillna(-99999,inplace = True) #sub nan with dummy data, since in ml, you dont want to lose data 


## generating data set for training, testing and predicting 
forecast_out = int(math.ceil(0.01*len(df)))  # number of days out 
print(forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

# X = np.array(df.drop(['label'],1)) # data uses captial X 
# y = np.array(df['label'])

# X = preprocessing.scale(X) # scale all values 
# y = np.array(df['label'])

X = np.array(df.drop(['label'],1)) # data feature captial X 
X = preprocessing.scale(X) # scale all values 
X = X[:-forecast_out]

X_lately = X[-forecast_out:] # we need to predict against, we dont have y value for 
df.dropna(inplace=True)
y = np.array(df[:-forecast_out]['label'])
# y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)


## generating classifier from sklearn and training it 
#can use multiple treads and parallel programming!! n_jobs yea!!!
# classifier= LinearRegression(n_jobs = -1) 
# classifier.fit(X_train,y_train) # actually training step 

# # save the already trained classifier so that it would be handy to use for later 
# with open('linearrepression.pickle','wb') as f:
# 	pickle.dump(classifier,f)
# # retrieve the classifier to use it 
pickle_in = open('linearrepression.pickle','rb') # -> if saved already, just use it
classifier = pickle.load(pickle_in)

accuracy = classifier.score(X_test,y_test)

# classifier= svm.SVR(kernel = 'poly') # what is a kernel 
# classifier.fit(X_train,y_train)
# accuracy = classifier.score(X_test,y_test)

### predict the classifier with unknown future data 
forecast_set = classifier.predict(X_lately) # predict unknow future values
print(forecast_set,accuracy,forecast_out)

df['Forest'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

## add timeframe back to theoriginal dataset 
for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i] #df.loc refers the index , next_date is a date stamp that is the index of the dataframe 

print(df.head()) # features at top
print(df.tail()) # predicted data at the end such that only foret is there

### plot the graph to visualize 
df['Adj. Close'].plot()
df['Forest'].plot()
plt.legend(loc = 4)
plt.xlabel('date')
plt.ylabel('price')
plt.show()



#print(accuracy)





