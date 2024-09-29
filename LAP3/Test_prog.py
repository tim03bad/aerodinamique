
#%%
from meanSquare import MeanSquare
import numpy as np
from meshGenerator import MeshGenerator
from meshConnectivity import MeshConnectivity
from mesh import Mesh

#%%
def fct(x,y):
    return 2*x + 3*y

def grad(x,y):

    return np.array([2,3])

#%%
param = {0: ('L', fct),1:('L',fct),2:('L',fct),3:('L',fct)}

#Erreur
E1 = 0

E2 = 0


#Aire moyenne
h1 = 0
h2 = 0


#%% Mesh 1
<<<<<<< HEAD
mesh_parameters1 = {'mesh_type': 'QUAD','Nx': 500, 'Ny': 500}
ms1 = MeanSquare(mesh_parameters1)
ms1.createCL(param)
ms1.setChamp(fct,grad)
ms1.createElements()
=======
mesh_parameters1 = {'mesh_type': 'TRI','lc':0.5}
mesher = MeshGenerator()
mesh_obj = mesher.rectangle([0.0, 1.0, 0.0, 1.0], mesh_parameters1)
conec = MeshConnectivity(mesh_obj)
conec.compute_connectivity()

ms1 = MeanSquare(mesh_obj,param,fct,grad)

>>>>>>> 63cddedb4bf3883df5e89745f0a89503ecc92a52
ms1.constructionATA()
ms1.constructionB()

<<<<<<< HEAD
# %% Mesh 2
mesh_parameters2 = {'mesh_type': 'QUAD','Nx': 600, 'Ny': 600}
ms2 = MeanSquare(mesh_parameters2)
ms2.createCL(param)
ms2.setChamp(fct,grad)
ms2.createElements()
=======
ms1.calculMeanSquare()
E1 = ms1.errorQuadratique()
h1 = ms1.calculTailleMoyenne()

print("h 1 : ",h1)
print("Error 1 : ",E1)


#%% Mesh 2
mesh_parameters2 = {'mesh_type': 'TRI','lc':0.08}
mesh_obj2 = mesher.rectangle([0.0, 1.0, 0.0, 1.0], mesh_parameters2)
conec = MeshConnectivity(mesh_obj2)
conec.compute_connectivity()

ms2 = MeanSquare(mesh_obj2,param,fct,grad)

>>>>>>> 63cddedb4bf3883df5e89745f0a89503ecc92a52
ms2.constructionATA()
ms2.constructionB()

<<<<<<< HEAD
#%% Ordre de convergence

print("Ordre de convergence : ", np.log(E2/E1)/np.log(h2/h1))
# %%
=======
ms2.calculMeanSquare()
E2 = ms2.errorQuadratique()
h2 = ms2.calculTailleMoyenne()

print("h 2 : ",h2)
print("Error 2 : ",E2)


#%%
print("Ordre de convergence : ", np.log(E1/E2)/np.log(h1/h2))


>>>>>>> 63cddedb4bf3883df5e89745f0a89503ecc92a52
