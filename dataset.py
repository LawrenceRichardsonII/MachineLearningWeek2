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
close_px.plot(label='AAPL')
#this was the dataframe manipulation
mavg.plot(label='mavg')
plt.legend()


plt.show()

