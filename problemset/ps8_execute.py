#!/usr/bin/env python
# coding: utf-8

# In[148]:


pip install ipynb


# In[170]:


# import function
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ipynb.fs.full.functions import utility 
sns.set()

get_ipython().run_line_magic('matplotlib', 'inline')


# In[101]:


# Step 1: Set parameters
beta = 0.95
sigma = 1.0
Rf=1.01
rho = 0.8
sigma_eps = 0.05


# In[102]:


# Step 2: Set Grid for State Space  
'''
------------------------------------------------------------------------
Create Grid for State Space    
------------------------------------------------------------------------
lb_w      = scalar, lower bound of initial wealth grid
ub_w      = scalar, upper bound of initial wealth grid 
size      = integer, number of grid points
w         = vector, size x 1 vector of initial wealth grid points 
eps       = vector, size x 1 vector of error term
y         = vector, size x 1 vector of income grid points 
Rm        = vector, size x 1 vector of market return grid points 
------------------------------------------------------------------------
'''
lb_w = 0.4 
ub_w = 2.0 
size = 20  # Number of grid points
w = np.linspace(lb_w, ub_w, size)
# y and Rm follow AR(1) process
eps = np.random.normal(0.0, sigma_eps, size=(size))
y = np.empty(size) # y<(0.1,1)
Rm = np.empty(size)
y[0] = 0.1 + eps[0]
Rm[0] = Rf + eps[0]
for i in range(1, size):
     y[i] = rho * y[i - 1] + (1 - rho) * 1 + eps[i]
     Rm[i] = rho * Rm[i - 1] + (1 - rho) * Rf + eps[i]


# In[105]:


# step 3: Create grid of current utility values  
'''
------------------------------------------------------------------------
Create grid of current utility values    
------------------------------------------------------------------------
Sm       = matrix, current investment in market asset 
Sf       = matrix, current investment in risk-free asset  (Sf=(w'-Rm*Sm)/Rf)
C        = matrix, current consumption (c=w+y-Sm-Sf)
U        = matrix, current period utility value for all possible
           choices of w,y,Rm and w' 
------------------------------------------------------------------------
'''
Sm = np.zeros((size, size,size,size,size)) 
Sf = np.zeros((size, size,size,size,size)) 
c = np.zeros((size, size,size,size,size)) 
for i in range(size): # loop over w
    for j in range(size):# loop over y
        for k in range(size):# loop over Rm
            for m in range(size):# loop over w'
               S=np.linspace(0,w[i]+y[j],size)
               Sm[i,j,k,m,:]=S
                
for i in range(size): # loop over w
    for j in range(size):# loop over y
        for k in range(size):# loop over Rm
            for m in range(size):# loop over w'
                for l in range(size):# loop over Sm
                    Sf[i,j,k,m,l] = (w[m]-Rm[k]*Sm[i,j,k,m,l])/Rf
                    if Sf[i,j,k,m,l]>0:
                        c[i,j,k,m,l]=w[i]+y[j]-Sm[i,j,k,m,l]-Sf[i,j,k,m,l]
                    else:
                        c[i,j,k,m,l]=1e-15
                        
# replace 0 and negative c with a tiny value 
# This is a way to impose non-negativity on c
c[c<=0] = 1e-15


# In[146]:


#step4:Value Function Iteration 
'''    
------------------------------------------------------------------------
VFtol     = scalar, tolerance required for value function to converge
VFdist    = scalar, distance between last two value functions
VFmaxiter = integer, maximum number of iterations for value function
V         = vector, the value functions at each iteration
Vmat      = matrix, the value for each possible combination of w and w'
VFiter    = integer, current iteration number
TV        = vector, the value function after applying the Bellman operator
PF_Sm     = vector, indicies of choices of Sm for all w 
PF_w      = vector, indicies of choices of w' for all w 
VF        = vector, the "true" value function
------------------------------------------------------------------------
'''
VFtol = 1e-8 
VFdist = 7.0 
VFmaxiter = 500
V = np.zeros((size,size,size)) # initial guess at value function
Vmat = np.zeros((size,size,size,size,size)) # initialize Vmat matrix
VFiter = 1 
while VFdist > VFtol and VFiter < VFmaxiter:  
    for i in range(size): # loop over w
        for j in range(size):# loop over y
            for k in range(size):# loop over Rm
                for m in range(size):# loop over w'
                    for l in range(size):# loop over Sm
                        Vmat[i,j,k,m,l] = utility(c[i,j,k,m,l],sigma) + beta * V[m,j,k] 
    
    Vmat_1 = Vmat.max(4)# apply max operator to Vmat (to get V(w,y,Rm))
    TV = Vmat_1.max(3)
    PF_Sm = np.argmax(Vmat, axis=4)
    PF_w = np.argmax(Vmat_1, axis=3)
    VFdist = (np.absolute(V - TV)).max()  # check distance
    V = TV
    VFiter += 1 

if VFiter < VFmaxiter:
    print('Value function converged after this many iterations:', VFiter)
else:
    print('Value function did not converge')            


VF = V # solution to the functional equation


# In[ ]:


# Step 5: Extract decision rules from solution
'''
------------------------------------------------------------------------
Find c,Sf,Sm policy functions   
------------------------------------------------------------------------
optW  = vector, the optimal choice of w'
optC  = vector, the optimal choice of c 
------------------------------------------------------------------------
'''
optSm_1 = Sm[PF_Sm[:,0,0,0]]
optSm=optSm_1[:,0,0,0,0]# today's optimal investment in market asset 
optW = w[PF_w[[:,0,0]]] # tomorrow's optimal wealth 
optSf=(optW-optSm*Rm)/Rf
optC=w+y-optSm-optSf


# In[129]:


# step 6: Visualize output
# Plot value function as a function of initial wealth
plt.figure()
plt.plot(w[1:], VF[1:,0,0])
plt.xlabel('initial wealth')
plt.ylabel('Value Function')
plt.title('Value Function - portfolio choice')
plt.show()


# In[136]:


#Plot optimal invesment in risk-free asset rule as a function of initial wealth
plt.figure()
fig, ax = plt.subplots()
ax.plot(w[1:], optSf[1:], label='invest in risk-free asset')
# Now add the legend with some customizations.
legend = ax.legend(loc='upper left', shadow=False)
# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')
for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
plt.xlabel('initial wealth')
plt.ylabel('invest in Rf')
plt.title('Policy Function, invest in Rf')
plt.show()


# In[144]:


#Plot optimal invesment in market asset as a function of initial wealth
plt.figure()
fig, ax = plt.subplots()
ax.plot(w[1:], optSm[1:], label='invest in market asset')
# Now add the legend with some customizations.
legend = ax.legend(loc='upper left', shadow=False)
# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')
for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
plt.xlabel('initial wealth')
plt.ylabel('invest in Rm')
plt.title('Policy Function, invest in Rm')
plt.show()


# In[131]:


#Plot optimal consumption as a function of initial wealth
plt.figure()
fig, ax = plt.subplots()
ax.plot(w[1:], optC[1:], label='consumption')
# Now add the legend with some customizations.
legend = ax.legend(loc='upper left', shadow=False)
# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')
for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
plt.xlabel('initial wealth')
plt.ylabel('consumption')
plt.title('Policy Function, consumption')
plt.show()


# In[134]:


#Plot optimal invesment in risk-free asset as a function of market return
plt.figure()
fig, ax = plt.subplots()
ax.plot(Rm[1:], optSf[1:], label='invest in risk-free asset')
# Now add the legend with some customizations.
legend = ax.legend(loc='upper left', shadow=False)
# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')
for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
plt.xlabel('market return')
plt.ylabel('invest in Rf')
plt.title('Policy Function, invest in Rf')
plt.show()


# In[ ]:




