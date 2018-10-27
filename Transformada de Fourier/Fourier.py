
# coding: utf-8

# ### Punto 2

# In[1]:


#Needed imports 
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


#read the data from the dat archive signal
signal = np.loadtxt("signal.dat",delimiter = ",")
incomplete = np.loadtxt("incompletos.dat",delimiter=",")


# In[7]:


#Separate the 2 columns of the data to make the plot 
signal_1 = signal[:,0]
signal_2 = signal[:,1]
plt.title("Signal")
plt.plot(signal_1,signal_2)
plt.grid(True)
plt.ylabel("Data in signal 1")
plt.xlabel("Data in signal 0")
plt.savefig("MartinezDiego_signal.pdf", type = "pdf")

