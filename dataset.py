import pandas as pd
import datetime
import pandas_datareader.data as web
import math
import numpy as np
from pandas import Series, DataFrame
from pandas.plotting import scatter_matrix

from sklearn import tree



import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline

start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2017, 1, 11)

df = web.DataReader("AAPL", 'yahoo', start, end)
#.tail shows only the end, we actually need to do a .print if we want to see in line on cmder
df.tail()

#construct our moving averages
close_px = df['Adj Close']
mavg = close_px.rolling(window=100).mean()

#format for matlabplot (to print and visulaize)
mpl.rc('figure', figsize=(8, 7))
mpl.__version__

# Adjusting the style of matplotlib
style.use('ggplot')

#this is our dataframe series we pulled in from the datareader, could also be from a csv, give it a label
#can only show one plt at a time
#close_px.plot(label='AAPL')

#this was the dataframe manipulation
#mavg.plot(label='mavg')



rets = close_px / close_px.shift(1) - 1

#rets.plot(label='return')

#pull more data into a new dataframe for correlation matrices
dfcomp = web.DataReader(['AAPL', 'GE', 'GOOG', 'IBM', 'MSFT'],'yahoo',start=start,end=end)['Adj Close']

#pandas functions here
retscomp = dfcomp.pct_change()
#need to print corr if we want to visualize in commandline
corr = retscomp.corr()

#scatterplot time

#plt.scatter(retscomp.AAPL, retscomp.GE)
#plt.xlabel('Returns AAPL')
#plt.ylabel('Returns GE')


#pd.plotting.scatter_matrix(retscomp, diagonal='kde', figsize=(10, 10));


#heatmap
#plt.imshow(corr, cmap='hot', interpolation='none')
#plt.colorbar()
#plt.xticks(range(len(corr)), corr.columns)
#plt.yticks(range(len(corr)), corr.columns);

#return rate and risk
#plt.scatter(retscomp.mean(), retscomp.std())
#plt.xlabel('Expected returns')
#plt.ylabel('Risk')
#for label, x, y in zip(retscomp.columns, retscomp.mean(), retscomp.std()):
#    plt.annotate(
#        label, 
#        xy = (x, y), xytext = (20, -20),
#        textcoords = 'offset points', ha = 'right', va = 'bottom',
#        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
#        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))


#building dataframe features
dfreg = df.loc[:,['Adj Close','Volume']]
dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0


##pre processing the data now
# Drop missing value
dfreg.fillna(value=-99999, inplace=True)
# We want to separate 1 percent of the data to forecast
forecast_out = int(math.ceil(0.01 * len(dfreg)))
# Separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'], 1))
# Scale the X so that everyone can have the same distribution for linear regression
X = preprocessing.scale(X)

# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
# Separate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]


# Linear regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X, y)
# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X, y)

# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X, y)

# KNN Regression
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X, y)

confidencereg = clfreg.score(X, y)
confidencepoly2 = clfpoly2.score(X, y)
confidencepoly3 = clfpoly3.score(X, y)
confidenceknn = clfknn.score(X, y)

# results
print('The linear regression confidence is ', confidencereg)
print('The quadratic regression 2 confidence is ', confidencepoly2)
print('The quadratic regression 3 confidence is ', confidencepoly3)
print('The knn regression confidence is ', confidenceknn)

#using clfknn here could be clfpoly etc
forecast_set = clfreg.predict(X_lately)
dfreg['Forecast'] = np.nan


#plotting our predicitons

last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

for i in forecast_set:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]
dfreg['Adj Close'].tail(500).plot()
dfreg['Forecast'].tail(500).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
