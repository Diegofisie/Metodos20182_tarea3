
# coding: utf-8

# In[38]:

#Importo lo necesario para realizar el ejercicio 
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg


# In[39]:

#Se importan los datos de la fuente WDBC sobre pacientes.
datos = np.genfromtxt("WDBC.dat", None, delimiter = "\n") 
datoscancer = np.zeros([len(datos),32])
prueba = (datos[0].decode('UTF-8')).split(',')


# In[45]:

#Se agregan los datos a una matriz 
for i in range(len(datos)):
    datoscancer[i] = (datos[i].decode('UTF-8')).split(',')


# In[46]:

datoscancer.shape


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



