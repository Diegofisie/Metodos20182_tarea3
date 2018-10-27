
# coding: utf-8

# ### Point 1

# In[6]:


#Needed imports
import numpy as np
import matplotlib.pyplot as plt


# In[7]:


#Read the data from WDBC.dat 
data = np.genfromtxt("WDBC.dat",None,delimiter="\n")


# In[8]:


#Create a vector called diagnsis wich represent the first value
diagnosis = np.zeros(len(data))
#Create a matriz for the other data 
otherdata = np.zeros([len(data),30])


# In[11]:


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

