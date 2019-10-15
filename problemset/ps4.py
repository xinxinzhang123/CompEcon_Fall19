#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages 
import pandas as pd
import numpy as np
import scipy.optimize as opt
from scipy.optimize import differential_evolution
from geopy.distance import geodesic


# In[2]:


# Read data
ps4 = pd.read_excel('radio_merger_data.xlsx')
#Part One
# create dataframe for counterfactual 
cf=np.array([])
# calculate number of observations in each year
a=[len(ps4[ps4.year==2007]),len(ps4[ps4.year==2008])]


# In[3]:


# find all counterfactuals
i=0
for i in range(0,2):
    j=0
    if i==1:
        m=a[0]
    else:
        m=0
    for j in range (0,a[i]-1):
        k=j+1
        for k in range (j+1,a[i]):
            # f(b,t)
            x1y1=ps4.iloc[j+m*i,9]*ps4.iloc[j+m*i,10]
            x2y1=ps4.iloc[j+m*i,11]*ps4.iloc[j+m*i,10]
            dis=geodesic((ps4.iloc[j+m*i,3],ps4.iloc[j+m*i,4]),(ps4.iloc[j+m*i,5],ps4.iloc[j+m*i,6])).miles
            # f(b`,t`)
            x1y1a=ps4.iloc[k+m*i,9]*ps4.iloc[k+m*i,10]
            x2y1a=ps4.iloc[k+m*i,11]*ps4.iloc[k+m*i,10]
            disa=geodesic((ps4.iloc[k+m*i,3],ps4.iloc[k+m*i,4]),(ps4.iloc[k+m*i,5],ps4.iloc[k+m*i,6])).miles
            # f(b`,t)
            x1y1b=ps4.iloc[j+m*i,9]*ps4.iloc[k+m*i,10]
            x2y1b=ps4.iloc[j+m*i,11]*ps4.iloc[k+m*i,10]
            disb=geodesic((ps4.iloc[j+m*i,3],ps4.iloc[j+m*i,4]),(ps4.iloc[k+m*i,5],ps4.iloc[k+m*i,6])).miles
            # f(b,t`)
            x1y1c=ps4.iloc[k+m*i,9]*ps4.iloc[j+m*i,10]
            x2y1c=ps4.iloc[k+m*i,11]*ps4.iloc[j+m*i,10]
            disc=geodesic((ps4.iloc[k+m*i,3],ps4.iloc[k+m*i,4]),(ps4.iloc[j+m*i,5],ps4.iloc[j+m*i,6])).miles
            # append to array cf
            cf=np.append(cf,[x1y1,x2y1,dis,x1y1a,x2y1a,disa,x1y1b,x2y1b,disb,x1y1c,x2y1c,disc])
            k=k+1
        j=j+1
    i=i+1


# In[4]:


# MSE functions
i=0
score=0
def mse(params):
    a = params[0]
    b = params[1]
    for i in range(0,2421):
        # f(b,t)+f(b`,t`)
        f1=cf[i*12]+a*cf[i*12+1]+b*cf[i*12+2]+cf[i*12+3]+a*cf[i*12+4]+b*cf[i*12+5]
        # f(b`,t)+f(b,t`)
        f2=cf[i*12+6]+a*cf[i*12+7]+b*cf[i*12+8]+cf[i*12+9]+a*cf[i*12+10]+b*cf[i*12+11]
        # f1>f2  
        s=(f1>f2)
        score=-s.sum()
        i=i+1
    return score


# In[5]:


bounds=[(-1,1),(-2,2)]
# minimum score
results = differential_evolution(mse,bounds)
results


# In[ ]:


#part 2


# In[6]:


# create dataframe for counterfactual 
cf=np.array([])
# calculate number of observations in each year
a=[len(ps4[ps4.year==2007]),len(ps4[ps4.year==2008])]
# find all counterfactuals
i=0
for i in range(0,2):
    j=0
    if i==1:
        m=a[0]
    else:
        m=0
    for j in range (0,a[i]-1):
        k=j+1
        for k in range (j+1,a[i]):
            # f(b,t)
            x1y1=ps4.iloc[j+m*i,9]*ps4.iloc[j+m*i,10]
            x2y1=ps4.iloc[j+m*i,11]*ps4.iloc[j+m*i,10]
            hhi=ps4.iloc[j+m*i,8]
            dis=geodesic((ps4.iloc[j+m*i,3],ps4.iloc[j+m*i,4]),(ps4.iloc[j+m*i,5],ps4.iloc[j+m*i,6])).miles
            # f(b`,t`)
            x1y1a=ps4.iloc[k+m*i,9]*ps4.iloc[k+m*i,10]
            x2y1a=ps4.iloc[k+m*i,11]*ps4.iloc[k+m*i,10]
            hhia=ps4.iloc[k+m*i,8]
            disa=geodesic((ps4.iloc[k+m*i,3],ps4.iloc[k+m*i,4]),(ps4.iloc[k+m*i,5],ps4.iloc[k+m*i,6])).miles
            # f(b,t`)
            x1y1b=ps4.iloc[j+m*i,9]*ps4.iloc[k+m*i,10]
            x2y1b=ps4.iloc[j+m*i,11]*ps4.iloc[k+m*i,10]
            hhib=ps4.iloc[k+m*i,8]
            disb=geodesic((ps4.iloc[j+m*i,3],ps4.iloc[j+m*i,4]),(ps4.iloc[k+m*i,5],ps4.iloc[k+m*i,6])).miles
            # f(b`,t)
            x1y1c=ps4.iloc[k+m*i,9]*ps4.iloc[j+m*i,10]
            x2y1c=ps4.iloc[k+m*i,11]*ps4.iloc[j+m*i,10]
            hhic=ps4.iloc[j+m*i,8]
            disc=geodesic((ps4.iloc[k+m*i,3],ps4.iloc[k+m*i,4]),(ps4.iloc[j+m*i,5],ps4.iloc[j+m*i,6])).miles
            # p(b,t) and p(b`,t`)
            p1=ps4.iloc[j+m*i,3]
            p2=ps4.iloc[k+m*i,3]
            # append to array cf
            cf=np.append(cf,[x1y1,x2y1,hhi,dis,x1y1a,x2y1a,hhia,disa,x1y1b,x2y1b,hhib,disb,x1y1c,x2y1c,hhic,disc,p1,p2])
            k=k+1
        j=j+1
    i=i+1


# In[256]:


i=0
m=18
score=0
def mse(params):
    a = params[0]
    b = params[1]
    c = params[2]
    d = params[3]
    for i in range(0,2421):
        # f(b,t)
        f1=a*cf[i*m]+b*cf[i*m+1]+c*cf[i*m+2]+d*cf[i*m+3]
        # f(b`,t`)
        f2=a*cf[i*m+4]+b*cf[i*m+5]+c*cf[i*m+6]+d*cf[i*m+7]
        # f(b`,t)
        f3=a*cf[i*m+8]+b*cf[i*m+9]+c*cf[i*m+10]+d*cf[i*m+11]
        # f(b,t`)
        f4=a*cf[i*m+12]+b*cf[i*m+13]+c*cf[i*m+14]+d*cf[i*m+15]
        # p1-p2
        p=cf[i*m+16]-cf[i*m+17]
           
        s=(f1-f4>p)&(f2-f3>-p)
        score=-s.sum()
        i=i+1
    return score


# In[7]:


bounds=[(-1,1),(-1,1),(-1,1),(-1,1)]
# minimum score
results = differential_evolution(mse,bounds)
results

