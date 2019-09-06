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
df.tail()

close_px = df['Adj Close']
mavg = close_px.rolling(window=100).mean()



# Adjusting the size of matplotlib

mpl.rc('figure', figsize=(8, 7))
mpl.__version__

# Adjusting the style of matplotlib
style.use('ggplot')

close_px.plot(label='AAPL')
mavg.plot(label='mavg')
plt.legend()


rets = close_px / close_px.shift(1) - 1
rets.plot(label='return')

dfcomp = web.DataReader(['AAPL', 'GE', 'GOOG', 'IBM', 'MSFT'],'yahoo',start=start,end=end)['Adj Close']

retscomp = dfcomp.pct_change()

corr = retscomp.corr()

plt.scatter(retscomp.AAPL, retscomp.GE)
plt.xlabel(‘Returns AAPL’)
plt.ylabel(‘Returns GE’)