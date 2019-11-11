#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas_datareader as pdr
from pandas_datareader import data as web
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import linearmodels as lm
from stargazer.stargazer import Stargazer
from linearmodels.panel import PanelOLS


# In[2]:


#import daily stock price of Google from Janurary 1st, 2017 to July 1st, 2019 from yahoo finance
ticker="GOOGL"
start_time = datetime.datetime(2017, 1, 1) 
end_time = datetime.datetime(2019, 7, 1)
data_source='yahoo'

dp=web.DataReader(ticker,data_source=data_source,start=start_time,end=end_time)


# In[3]:


#create daily returns
dp["close_lagged"] = dp.Close.shift(1)
dp["dr"]=(-dp["close_lagged"]+dp["Close"])/dp["close_lagged"]*100


# In[4]:


#import daily risk factors from Fama_French database
df = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench",start=start_time,end=end_time)[0]


# In[5]:


#merge two database based on index
ps6=pd.merge(df,dp,left_index=True,right_index=True)


# In[6]:


# create excess daily return
ps6["exdr"]=ps6["dr"]-ps6["RF"]
ps6["mkt"]=ps6["Mkt-RF"]
ps6=ps6.dropna()


# In[7]:


# Histograms
# draw the distribution of daily excess return
plt.style.use('ggplot') 
sns.distplot(ps6['exdr'], kde=True, rug=False)
plt.title('Distribution of daily excess return')
# daily excess return seems to follow normal distribution


# In[22]:


# test CAPM
# regress daily excess return on daily risk factors
OLS = smf.ols(formula='exdr ~ mkt+SMB+HML', data=ps6)
# I use panel data to regression daily excess return on common risk factors (mkt_rf,smb,hml).
res = OLS.fit()
print(res.summary())


# In[ ]:




