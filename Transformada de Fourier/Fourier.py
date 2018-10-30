
# coding: utf-8

# ### Punto 2

# In[2]:


#Needed imports 
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
from scipy.fftpack import ifft
from scipy import interpolate


# In[3]:


#read the data from the dat archive signal
signal = np.loadtxt("signal.dat",delimiter = ",")
incomplete = np.loadtxt("incompletos.dat",delimiter=",")


# In[4]:


#Separate the 2 columns of the data to make the plot 
signal_1 = signal[:,0]
signal_2 = signal[:,1]
plt.title("Signal")
plt.plot(signal_1,signal_2)
plt.grid(True)
plt.ylabel("Data in signal 1")
plt.xlabel("Data in signal 0")
plt.savefig("MartinezDiego_signal.pdf", type = "PDF")


# In[5]:


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


# In[6]:


#Fourier transform of the data and transformation of the freq
Transf=fourier(signal_2)
n = len(Transf)
pf = (1/(signal_1[1]-signal_1[0]))
freq = np.concatenate((np.linspace(0,pf,len(signal_2))[0:256],np.linspace(-pf,0,len(signal_2))[255:511]))
print("No se utilizo el paquete de fftfreq :)")


# In[7]:


print("Las frecuencias principales son:")
print(freq[Transf>100])


# In[8]:


plt.title("Grafica de la transformada")
plt.plot(freq,np.abs(Transf),label="Transformada")
plt.legend(loc = 0)
plt.ylabel("Amplitud")
plt.xlabel("Frecuencia[Hz]")
plt.xlim([-1000,1000])
plt.grid(True)
plt.savefig("MartinezDiego_TF.pdf", type = "PDF")


# In[9]:


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

# In[10]:


incomp_1 = incomplete[:,0]
incomp_2 = incomplete[:,1]
plt.plot(imcomp_1,incomp_2)
plt.title("Datos incompletos")
plt.ylabel("Senal")
plt.xlabel("Tiempo")
plt.grid(True)
print("No se puede encontrar la tranformada de los datos ya que estos tienen una tasa de muestreo muy pequena y esto hace que no hayan suficientes datos como para recontruir la ransformada")


# In[ ]:


#Interpolation on the data, quadratic and cuvis to find the transform.
fquadratic = interpolate.interp1d(incomp_1,incomp_2,kind = "quadratic")
fcubic = interpolate.interp1d(incomp_1,incomp_2,kind = "cubic")
data_x = np.linspace(imcomp_1[0],imcomp_1[-1],512)
datafcuadra = fquadratic(data_x)
datafcubic = fcubic(data_x)
transform=fourier(incomp_2)
tranformquadratic=fourier(datafcuadra)
tranformcubic=fourier(datafcubic)
pf = (1/(data_x[1]-data_x[0]))
freqO = np.concatenate((np.linspace(0,pf,len(imcomp_1))[0:256],np.linspace(-pf,0,len(imcomp_1))[255:511]))
freq = np.concatenate((np.linspace(0,pf,len(data_x))[0:256],np.linspace(-pf,0,len(data_x))[255:511]))


# In[ ]:


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
plt.close()


# In[ ]:


#Diferences btw the graphs.
print("Como se puede ver en las graficas cuando se realiza la interpolacion de los datos se puede ver un aumento en la cantidad de los datos lo cual ayuda a tenr una transformada de fourier efectiva ")


# In[ ]:


#filters
# 1000Hz
fc=1000
transform[abs(freqO)>fc]=0
tiempo_0=ifft(transform)
tranformquadratic[abs(freq)>fc]=0
tiempo_qua=ifft(tranformquadratic)
tranformcubic[abs(freq)>fc]=0
tiempo_cubic=ifft(tranformcubic)
# 500Hz
fc=500
transform[abs(freqO)>fc]=0
tiempo_0_2=ifft(transform)
tranformquadratic[abs(freq)>fc]=0
tiempo_qua_2=ifft(tranformquadratic)
tranformcubic[abs(freq)>fc]=0
tiempo_cubic_2=ifft(tranformcubic)


# In[ ]:


#Plot of the interpoltions filtred
plt.figure()
plt.subplot(321)
plt.plot(incomp_1,np.real(tiempo_0),label="Original 1000Hz ")
plt.ylabel("Amplitud real")
plt.legend()

plt.subplot(322)
plt.plot(incomp_1,np.real(tiempo_0_2),label="Original 500Hz ")
plt.ylabel("Amplitud real")
plt.legend()


plt.subplot(323)
plt.plot(data_x,np.real(tiempo_qua),label="Quadratic 1000Hz")
plt.ylabel("Amplitud real")
plt.legend()

plt.subplot(324)
plt.plot(data_x,np.real(tiempo_qua_2),label="Cubic 500Hz")
plt.legend()

plt.subplot(325)
plt.plot(data_x,np.real(tiempo_cubic),label="Cubic 1000Hz")
plt.legend()
plt.ylabel("Amplitud real")
plt.xlabel("Frecuencia[Hz]")

plt.subplot(326)
plt.plot(data_x,np.real(tiempo_cubic_2),label="Cubic 500Hz")
plt.xlabel("Frecuencia[Hz]")
plt.legend()
plt.savefig("MartinezDiego_2Filtros.pdf",type = "pdf")

