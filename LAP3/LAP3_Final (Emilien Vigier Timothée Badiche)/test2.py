
#%%
from meanSquare import MeanSquare
import numpy as np
from meshGenerator import MeshGenerator
from meshConnectivity import MeshConnectivity
from mesh import Mesh
from meshPlotter import MeshPlotter

#%%
def fct(x,y):
    return 100*x + 0*y + 100

def grad(x,y):

    return np.array([100,0])

#%%
param = {0: ('D', 100),1:('N',0),2:('D',200),3:('N',0)}

#Erreur
E1 = 0



#Aire moyenne
h1 = 0



#%% Mesh 1
mesh_parameters1 = {'mesh_type': 'QUAD','Nx':5,'Ny':5}
mesher = MeshGenerator()
mesh_obj = mesher.rectangle([0.0, 1.0, 0.0, 1.0], mesh_parameters1)
conec = MeshConnectivity(mesh_obj)
conec.compute_connectivity()

#%%
plotter = MeshPlotter()
plotter.plot_mesh(mesh_obj, label_points=True, label_elements=True, label_faces=True)


#%%
ms1 = MeanSquare(mesh_obj,param,fct,grad)

ms1.constructionATA()
ms1.constructionB()

ms1.calculMeanSquare()
E1 = ms1.errorQuadratique()
h1 = ms1.calculTailleMoyenne()

print("h 1 : ",h1)
print("Error 1 : ",E1)