
# coding: utf-8

# In[353]:


import numpy as np
import matplotlib.pyplot as plt


# In[354]:


def funcion_hermite(x):
    return np.exp(-x**2)
#definición de la función de los polinomios
def polinomios_de_hermite(x,n):
    #Para el primer polinomio con n=0, se tiene que el resultado será la multifilicaciones de las funciones
    if n == 0:
        valor_polinomio = ((-1)**n*np.exp(x**2)*np.exp(-x**2))[0:-1]
    #Para el segundo polinomio n=1, se tiene que el resutado es la función y la primera devivada
    elif n == 1:
        def primera_derivada(f):
            h = x[1]-x[0]
            derivada = (f[1:] - f[0:-1])/h
            return derivada
        primera_derivada = primera_derivada(funcion_hermite(x))
        valor_polinomio = primera_derivada*np.exp((x[0:-1])**2)*(-1)**n
        #Para el resto de los polinimios se utiliza recursividad. como son derivadas de exponenciales, las siguientes derivadas
        #solo serán una combinacion de las anteriores.
        #Para eso se tiene que H_n+1(x) = 2xH_n(x) - 2(n)Hn-1(x)
    else:
        valor_polinomio = 2*x[0:-1]*polinomios_de_hermite(x,n-1) - 2*(n-1)*polinomios_de_hermite(x,n-2)
    return valor_polinomio


# In[355]:


x = np.linspace(-1,1,1000)
primer_polinomio = polinomios_de_hermite(x[0:-1],0)
segundo_polinomio = polinomios_de_hermite(x[0:-1],1)
tercer_polinomio = polinomios_de_hermite(x[0:-1],2)
cuarto_polinomio = polinomios_de_hermite(x[0:-1],3)
quinto_polinomio = polinomios_de_hermite(x[0:-1],4)
sexto_polinomio = polinomios_de_hermite(x[0:-1],5)
septimo_polinomio = polinomios_de_hermite(x[0:-1],6)


# In[356]:


plt.title('Polinomios de Hermite')
plt.plot(x[1:-1],primer_polinomio, label = 'n = 0')
plt.plot(x[1:-1],segundo_polinomio, label = 'n = 1')
plt.plot(x[1:-1],tercer_polinomio, label = 'n  = 2')
plt.plot(x[1:-1],cuarto_polinomio, label = 'n = 3')
plt.plot(x[1:-1],quinto_polinomio, label = 'n = 4')
plt.plot(x[1:-1],sexto_polinomio, label = 'n = 5')
plt.plot(x[1:-1],septimo_polinomio,label = 'n = 6')
plt.legend(loc = 0)
plt.grid(True)
plt.savefig("MartinezDiego_Hermite.pdf",format="pdf")
plt.close()


# In[357]:


x = np.linspace(-1,1,10000)
primer_polinomio = polinomios_de_hermite(x[0:-1],0)
segundo_polinomio = polinomios_de_hermite(x[0:-1],1)
tercer_polinomio = polinomios_de_hermite(x[0:-1],2)
cuarto_polinomio = polinomios_de_hermite(x[0:-1],3)
quinto_polinomio = polinomios_de_hermite(x[0:-1],4)
sexto_polinomio = polinomios_de_hermite(x[0:-1],5)
septimo_polinomio = polinomios_de_hermite(x[0:-1],6)
plt.plot(x[1:-1],primer_polinomio, label = 'n = 0')
plt.legend(loc = 0)
plt.plot(x[1:-1],segundo_polinomio, label = 'n = 1')
plt.legend(loc=0)
plt.plot(x[1:-1],tercer_polinomio, label = 'n  = 2')
plt.legend(loc = 0)
plt.plot(x[1:-1],cuarto_polinomio, label = 'n = 3')
plt.legend(loc = 0)
plt.plot(x[1:-1],quinto_polinomio, label = 'n = 4')
plt.legend(loc = 0)
plt.plot(x[1:-1],sexto_polinomio, label = 'n = 5')
plt.legend(loc = 0)
plt.plot(x[1:-1],septimo_polinomio,label = 'n = 6')
plt.legend(loc = 0)
plt.grid(True)
plt.title('Polinomios de Hermite, 10 veces mas puntos')
plt.savefig("MartinezDiego_Hermite_10.pdf",format="pdf")
plt.close()

