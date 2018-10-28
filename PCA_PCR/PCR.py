
# coding: utf-8

# ### Point 1

# In[2]:


#Needed imports
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


#Read the data from WDBC.dat 
data = np.genfromtxt("WDBC.dat",None,delimiter="\n")


# In[4]:


#Create a vector called diagnsis wich represent the first value
diagnosis = np.zeros(len(data))
#Create a matriz for the other data 
otherdata = np.zeros([len(data),30])


# In[6]:


#built the matrix of the data in WDBC.dat by adding each row
for i in range(len(data)):
    row = (data[i].decode('UTF-8')).split(",")
    #Give values of the first vector, if it is Malign of Benig. 1 for M and 0 for B
    if row[1]=='M':
        diagnosis[i]=1
    elif row[1]=='B':
        diagnosis[i]=0
    #Keep building the matrix
    for j in range(30):
        otherdata[i][j]=(row)[j+2]
Matrix = np.transpose(otherdata)
#Normalization of the matrix
for i in range(len(Matrix)):
    Matrix[i]=Matrix[i]/np.std(Matrix[i])


# In[14]:


#To create the covariance matrix beetwen the data
#function that gives the covariance bt two vectors 
def covar(x,y):
    meanx = np.mean(x)
    meany = np.mean(y)
    return np.sum((x-meanx)*(y-meany))/(len(x)-1)
cov = np.empty([len(Matrix),len(Matrix)])
for i in range(len(Matrix)):
    for j in range(len(Matrix)):
        cov[i,j]=covar(Matrix[:,j],Matrix[:,i])
print(cov)

