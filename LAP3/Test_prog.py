
#%%
from meanSquare import MeanSquare
import numpy as np

#%%
def fct(x,y):
    return 2*x + 3*y

def grad(x,y):

    return np.array([2,3])

#%%
param = {0: ('N', grad),1:('N',grad),2:('N',grad),3:('N',grad)}

#Erreur
E1 = 0

E2 = 0


#Aire moyenne
h1 = 0
h2 = 0


#%% Mesh 1
mesh_parameters1 = {'mesh_type': 'TRI','lc':0.5}
ms1 = MeanSquare(mesh_parameters1)
ms1.createCL(param)

ms1.setChamp(fct,grad)
ms1.createElements()
ms1.constructionATA()
ms1.constructionB()
ms1.calculMeanSquare()
E1 = ms1.error()
h1 = ms1.calculTailleMoyenne()
print("h : ",h1)
print("Error : ",E1)

# %% Mesh 2
mesh_parameters2 = {'mesh_type': 'TRI','lc':0.5}
ms2 = MeanSquare(mesh_parameters2)
ms2.createCL(param)

ms2.setChamp(fct,grad)
ms2.createElements()
ms2.constructionATA()
ms2.constructionB()
ms2.calculMeanSquare()
E2 = ms2.error()
h2 = ms2.calculTailleMoyenne()
print("h : ",h2)
print("Error : ",E2)

#%% Ordre de convergence
print("Ordre de convergence : ", np.log(E1/E2)/np.log(h1/h2))
# %%
