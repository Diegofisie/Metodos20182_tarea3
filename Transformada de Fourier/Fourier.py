
# coding: utf-8

# ### Punto 2

# In[26]:


#Needed imports 
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
from scipy.fftpack import ifft
from scipy import interpolate


# In[27]:


#read the data from the dat archive signal
signal = np.loadtxt("signal.dat",delimiter = ",")
incomplete = np.loadtxt("incompletos.dat",delimiter=",")


# In[28]:


#Separate the 2 columns of the data to make the plot 
signal_1 = signal[:,0]
signal_2 = signal[:,1]
plt.title("Signal")
plt.plot(signal_1,signal_2)
plt.grid(True)
plt.ylabel("Data in signal 1")
plt.xlabel("Data in signal 0")
plt.savefig("MartinezDiego_signal.pdf", type = "PDF")


# In[29]:


#Implementation of fourier tranform
def fourier(signal_2):
    N = len(signal_2)
    transform = np.zeros([len(signal_2)],dtype=complex)
    expo = np.exp(-2j*np.pi/N)
    for n in range(N):
        valorn = 0*np.exp(1j)
        for k in range(N):
            valorn+=(expo**(n*k))*signal_2[k]
        transform [n]=valorn
    return transform 


# In[34]:


#Fourier transform of the data and transformation of the freq
Transf=fourier(signal_2)
n = len(Transf)
pf = (1/(signal_1[1]-signal_1[0]))
freq = np.concatenate((np.linspace(0,pf,len(signal_2))[0:256],np.linspace(-pf,0,len(signal_2))[255:511]))
print("No se utilizo el paquete de fftfreq :)")


# In[35]:


plt.title("Grafica de la transformada")
plt.plot(freq,np.abs(Transf),label="Transformada")
plt.legend(loc = 0)
plt.ylabel("Amplitud")
plt.xlabel("Frecuencia[Hz]")
plt.xlim([-1000,1000])
plt.grid(True)
plt.savefig("MartinezDiego_TF.pdf", type = "PDF")


# In[36]:


#lowpass filter
fc=1000
Transf[abs(freq)>fc]=0
Tiempo=ifft(Transf)
plt.figure()
plt.grid(True)
plt.plot(signal_1,np.real(Tiempo))
plt.title("Datos filtrados")
plt.ylabel("Senial")
plt.xlabel("Tiempo")
plt.savefig("MartinezDiego_filtrada.pdf",type = "pdf")


# ### Datos incompletos

# In[43]:


incomp_1 = incomplete[:,0]
incomp_2 = incomplete[:,1]
plt.plot(imcomp_1,incomp_2)
plt.title("Datos incompletos")
plt.ylabel("Senal")
plt.xlabel("Tiempo")
plt.grid(True)
print("No se puede encontrar la tranformada de los datos ya que estos tienen una tasa de muestreo muy pequena y esto hace que no hayan suficientes datos como para recontruir la ransformada")


# In[49]:


#Interpolation on the data, quadratic and cuvis to find the transform.
fquadratic = interpolate.interp1d(incomp_1,incomp_2,kind = "quadratic")
fcubic = interpolate.interp1d(incomp_1,incomp_2,kind = "cubic")
data_x = np.linspace(imcomp_1[0],imcomp_1[-1],512)
datafcuadra = fquadratic(data_X)
datafcubic = fcubic(data_X)
transform=fourier(incomp_2)
tranformquadratic=fourier(datafcuadra)
tranformcubic=fourier(datafcubic)
pf = (1/(data_x[1]-data_X[0]))
freqO = np.concatenate((np.linspace(0,pf,len(imcomp_1))[0:256],np.linspace(-pf,0,len(imcomp_1))[255:511]))
freq = np.concatenate((np.linspace(0,pf,len(data_x))[0:256],np.linspace(-pf,0,len(data_X))[255:511]))


# In[53]:


plt.figure()
plt.subplot(221)
plt.plot(freqO,transform,label="trans Original")
plt.legend()
plt.grid(True)
plt.ylabel("Amplitud")
plt.xlabel("Frecuencia[Hz]")
plt.subplot(222)
plt.plot(freq,tranformquadratic,label="Quadratic" )
plt.legend()
plt.grid(True)
plt.ylabel("Amplitud")
plt.xlabel("Frecuencia[Hz]")
plt.subplot(223)
plt.plot(freq,tranformcubic,label="Cubic")
plt.legend()
plt.grid(True)
plt.ylabel("Amplitud")
plt.xlabel("Frecuencia[Hz]")
plt.savefig("MartinezDiego_TF_interpola.pdf",type = "pdf")

