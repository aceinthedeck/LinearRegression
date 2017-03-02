import pandas as pd
 
import matplotlib.pyplot as plt

#import scikit learn linear regression library
from sklearn.linear_model import LinearRegression


 
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
data.head()


fig,axs=plt.subplots(1,3,sharey=True)
data.plot(kind='scatter',x='TV',y='Sales',ax=axs[0],figsize=(16,8))
data.plot(kind='scatter',x='Radio',y='Sales',ax=axs[1])
data.plot(kind='scatter',x='Newspaper',y='Sales',ax=axs[2])


#x= TV advertising, y=sales
featureCols=['TV']
X=data[featureCols]
y=data.Sales
 
#build the model
lr=LinearRegression()
lr.fit(X,y)
 
#print the parameters
print(lr.intercept_)
print(lr.coef_)

featureCols=['Radio']
X=data[featureCols]
y=data.Sales
 
lr=LinearRegression()
lr.fit(X,y)
 
print(lr.intercept_)
print(lr.coef_)

#prints prediction for x=0
print(lr.predict(0))
#prints prediction for x=10
print(lr.predict(10))