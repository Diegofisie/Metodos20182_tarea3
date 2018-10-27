
# coding: utf-8

# ### Punto 2

# In[3]:


#Needed imports 
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft


# In[4]:


#read the data from the dat archive signal
signal = np.loadtxt("signal.dat",delimiter = ",")
incomplete = np.loadtxt("incompletos.dat",delimiter=",")


# In[18]:


#Separate the 2 columns of the data to make the plot 
signal_1 = signal[:,0]
signal_2 = signal[:,1]
plt.title("Signal")
plt.plot(signal_1,signal_2)
plt.grid(True)
plt.ylabel("Data in signal 1")
plt.xlabel("Data in signal 0")
plt.savefig("MartinezDiego_signal.pdf", type = "PDF")


# In[19]:


#Implementation of fourier tranform
def fourier(t):
    n=len(t)
    x=[]
    for k in range(0,n): #the fourier transform as the sumatory of the ak coeff of the fourier series
        sum=0
        for i in range (0,n):
            sum =  sum + t[i]*np.exp(-1j*2*np.pi*(k*i)/n)
        x.append(sum)
    return x


# In[20]:


#Fourier transform of the data and transformation of the freq
Transf=fourier(signal_2)
n = len(Transf)
freq = fft.fftfreq(n)
fixedfreq=2*np.pi*np.linspace(-n,n,n)/n
t=np.linspace(0,n,n)


# In[21]:


plt.title("Grafica de la transformada")
plt.plot(t,np.abs(Transf),label="Transformada")
plt.legend(loc = 0)
plt.grid(True)
plt.savefig("MartinezDiego_TF.pdf", type = "PDF")

