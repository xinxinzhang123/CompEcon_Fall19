#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import linearmodels as lm
from stargazer.stargazer import Stargazer
from linearmodels.panel import PanelOLS


# In[58]:


# Read in data from stata 
ps5 = pd.read_stata('regression.dta')


# In[19]:


#Part A
# 1.Histograms
# draw the distribution of holding period return
plt.style.use('ggplot') 
sns.distplot(ps5['lnhpr'], kde=True, rug=False)
plt.title('Distribution of holding period return')
# holding period return seems to follow normal distribution


# In[22]:


# 2.Bar plot by type (corporate bond or CLO)
# draw bar plot of holding period return by type
bar = pd.DataFrame({'Holing period return' : ps5.groupby('type').
                         apply(lambda x: np.average(x['lnhpr']))})
plt.style.use('ggplot') 
bar.plot(kind='bar', y="Holing period return", use_index=True)
plt.ylabel('Holing period return')
plt.xlabel('Type')
plt.title('Holing period return by type')
# CLO has higher holding period return than corporate bond


# In[17]:


# 3.scatter plots
# draw scatter plots of holding period return against market excess return by type
sns.pairplot(x_vars=['tmkt_rf'], y_vars=['lnhpr'], hue="type",data=ps5)
# CLO has much more positive holding period return than corporate bonds


# In[41]:


#Part B
# 1. OLS without fixed effect
hpr_OLS = smf.ols(formula='lnhpr ~ clo+tmkt_rf+tsmb+thml+tterm+tdef+hp', data=ps5)
# I use panel data to regression holding period return on common risk factors (tmkt_rf,tsmb,thml,tterm,and tdef) and 
# holding period. CLO is an indicator which is 1 if bond is CLO. If CLO is significant and positive, CLO has higher 
# return than corporate bond.
res = hpr_OLS.fit()
print(res.summary())
# The significant positive coefficient for CLO shows that CLO has higher excess return than corporate bond


# In[59]:


# 2. OLS with firm fixed effect
startyear = pd.Categorical(ps5.startyear)
ps5 = ps5.set_index(['entity_name','startyear'])


# In[67]:


exog_vars = ['clo','tmkt_rf','tsmb','thml','tterm','tdef','hp']
exog = sm.add_constant(ps5[exog_vars])
mod = PanelOLS(ps5.lnhpr, exog, entity_effects=True)
res = mod.fit()
print(res)
# After adding firm fixed effect, the coefficient of CLO is still significant positive and at similiar magnititude. 
# The argument that CLO has higher excess return than corporate return is valid.

