
# coding: utf-8

# ### Point 1

# In[3]:


#Needed imports
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


#Read the data from WDBC.dat 
data = np.genfromtxt("WDBC.dat",None,delimiter="\n")


# In[5]:


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


# In[7]:


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


# In[8]:


#computation of the eigenvalues and eigenvectors of the covar matrix
aguapanela = np.linalg.eig(cov)
eigvals = aguapanela[0]
eigvecs = aguapanela[1]
print("EigenValores: \n" + str(eigvals))
print("EigenVectores: \n" + str(eigvecs))


# In[9]:



print("las 2 variables principales son las 2 primeras, exeptuando el id y el B o M ya que para estas 2 variables la magnitud de los auto valores son las mas grandes")


# In[11]:


#PTo verefy the pca we need to evaluate the point product 
#To bening data
Ben1,Ben2=np.dot([eigvecs[0],eigvecs[1]],np.transpose(otherdata[diagnosis==0]))
#Datos Maligno
Mal1,Mal2=np.dot([eigvecs[0],eigvecs[1]],np.transpose(otherdata[diagnosis==1]))


# In[19]:


plt.figure()
plt.scatter(Ben1,Ben2,label="Benigno")
plt.scatter(Mal1,Mal2,label="Maligno")
plt.legend()
plt.grid(True)
plt.ylabel("PCABenigno")
plt.xlabel("PCAMaligno")
plt.savefig("MartinezDiego_PCA.pdf",type = "pdf")
print("De acuerdo con la gráfica que se obtuvo en el PCA se puede considerar una manera util de poder diagnostricar a un paciente, eso se da gracias a que las tendendias de dispecion hacia un lado o hace el otro demustran la suseptibilidad de un paciente a que su tomor sea maligno, dependiendo de donse se encuentre su lugar de acuerdo a sus datos.")

