import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame


import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl

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

#plt.legend()

rets = close_px / close_px.shift(1) - 1

rets.plot(label='return')

#pull more data into a new dataframe for correlation matrices
dfcomp = web.DataReader(['AAPL', 'GE', 'GOOG', 'IBM', 'MSFT'],'yahoo',start=start,end=end)['Adj Close']

#pandas functions here
retscomp = dfcomp.pct_change()
#need to print corr if we want to visualize in commandline
corr = retscomp.corr()

print (corr)


