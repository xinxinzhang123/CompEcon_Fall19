#!/usr/bin/env python
# coding: utf-8

# In[13]:


def utility(c,sigma):
    """
    Per period utility function
    """
    if sigma == 1:
        U = np.log(c)
    else:
        U = (c ** (1 - sigma)) / (1 - sigma)
    if c<=0:
        U = -99999
    return U  

