
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
param = {0: ('D', fct),1:('D',fct),2:('D',fct),3:('D',fct)}

#Erreur
E1 = 0

E2 = 0


#Aire moyenne
h1 = 0
h2 = 0


#%% Mesh 1
mesh_parameters1 = {'mesh_type': 'TRI','lc':0.5}
mesher = MeshGenerator()
mesh_obj = mesher.rectangle([0.0, 1.0, 0.0, 1.0], mesh_parameters1)
conec = MeshConnectivity(mesh_obj)
conec.compute_connectivity()

ms1 = MeanSquare(mesh_obj,param,fct,grad)

ms1.constructionATA()
ms1.constructionB()

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

ms2.constructionATA()
ms2.constructionB()

ms2.calculMeanSquare()
E2 = ms2.errorQuadratique()
h2 = ms2.calculTailleMoyenne()

print("h 2 : ",h2)
print("Error 2 : ",E2)


#%%
print("Ordre de convergence : ", np.log(E1/E2)/np.log(h1/h2))

input("Input to close...")


