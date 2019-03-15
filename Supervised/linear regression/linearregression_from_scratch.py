# writing own linear regression classifier 
from statistics import mean 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import style
import random 

# best_fit_line :
# y = mx + b 
# m = (mean(x)*mean(y) - mean(xy))/ (mean(x)*mean(x) - mean(x*x))
# b = (mean(y) - m*mean(x))
# determinator : r^2 = 1- se(best_fit(y))/se(mean(y))

# xs = np.array([1,2,3,4,5,6],dtype = np.float64)
# ys = np.array([5,4,6,5,6,7],dtype = np.float64)

style.use('fivethirtyeight')
# visualize 
# plt.scatter(xs,ys)
# plt.show()

# create testing dataset 
def create_dataset(count,variance,step = 2, correlation = False):
	val = 1
	ys = []
	for i in range(count):
		y = val + random.randrange(-variance,variance)
		ys.append(y)
		if correlation and correlation == 'pos':
			val += step
		elif correlation and correlation == 'neg':
			val -= step
	xs = [i for i in range(len(ys))]
	return np.array(xs,dtype=np.float64), np.array(ys,dtype = np.float64)

# best fit line 
def best_fit_slope(xs, yx):
	m = ((mean(xs)*mean(ys)) - mean(xs*ys)) / ((mean(xs)*mean(xs)) - mean(xs*xs)) # to get the power of 2, use **2 . ^2 does not work on float64 
	return m 

def best_fit_interception(xs,m,ys):
	b = mean(ys) - m *mean(xs)
	return b 

# coefficient determinator
def squared_error(y_line,y_orgin):
	return sum((y_orgin-y_line)**2)

def determinator(best_fit_e,mean_y_e):
	return 1-(best_fit_e/mean_y_e)


xs,ys = create_dataset(40,10,2,correlation='neg')


m = best_fit_slope(xs,ys)
b = best_fit_interception(xs,m,ys)
print(m)
print(b)

regression_line = [(m*x)+b for x in xs] 
mean_ys = [mean(ys) for x in xs]

squared_error_best = squared_error(regression_line,ys)
squared_error_mean = squared_error(mean_ys,ys)

determinator_ =determinator(squared_error_best,squared_error_mean)
print(determinator_)

# to predict unknown
predict_x = 8
predict_y = m*predict_x + b 


# visualize 
plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y,s = 100,color = 'g')
plt.plot(xs,regression_line)
plt.show()


# testing 


