
#%%
from meanSquare import MeanSquare
import numpy as np

#%%
def fct(x,y):
    return np.sin(x)+np.cos(y)

def grad(x,y):
    return np.array([np.cos(x),-np.sin(y)])

#%%
param = {0: ('D', fct),1:('D',fct),2:('D',fct),3:('D',fct)}

E1 =0
E2 =0

h1 =0
h2 =0


#%% Mesh 1
mesh_parameters1 = {'mesh_type': 'TRI','lc': 0.05}
ms1 = MeanSquare(mesh_parameters1)
ms1.createCL(param)
ms1.setChamp(fct,grad)
ms1.createElements()
ms1.constructionATA()
ms1.constructionB()
ms1.calculMeanSquare()
E1 = ms1.error()
h1 = ms1.calculTailleMoyenne()
print("Error : ",ms1.error())

# %% Mesh 2
mesh_parameters2 = {'mesh_type': 'TRI','lc': 0.01}
ms2 = MeanSquare(mesh_parameters2)
ms2.createCL(param)
ms2.setChamp(fct,grad)
ms2.createElements()
ms2.constructionATA()
ms2.constructionB()
ms2.calculMeanSquare()
E2 = ms2.error()
h2 = ms2.calculTailleMoyenne()
print("Error : ",ms2.error())

#%% Ordre de convergence
print("Ordre de convergence : ", np.log(E2/E1)/np.log(h2/h1))