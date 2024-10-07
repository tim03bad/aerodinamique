#%% Import
import numpy as np


from mesh import Mesh
from meshConnectivity import MeshConnectivity
from meshGenerator import MeshGenerator
from meshPlotter import MeshPlotter

from meanSquare import MeanSquare
from element import Element



from solver import Solver
import matplotlib.pyplot as plt

#%% Geométrie du maillage et donnée

Lx = 1
Ly = 1
Gamma = 0.5
q = 0
Cp = 1
rho = 1


def velocityField(x,y):
    return np.array([0.5,0])

def SolAnalytique(x,y):
    return (100/Lx + (q/(2*Gamma))*(Lx-x))*x + 100 + 0*y


mesh_params_Q1 = {
    'mesh_type': 'TRI',
    'lc': 0.04
}

mesh_params_Q2 = {
    'mesh_type': 'TRI',
    'lc': 0.01
}

mesher = MeshGenerator(verbose=True)
mesh_objQ1 = mesher.rectangle([0, Lx, 0, Ly], mesh_params_Q1)
conec = MeshConnectivity(mesh_objQ1)
conec.compute_connectivity()


#%%

mesher = MeshGenerator(verbose=True)
mesh_objQ2 = mesher.rectangle([0, Lx, 0, Ly], mesh_params_Q2)
conec = MeshConnectivity(mesh_objQ2)
conec.compute_connectivity()


#%%
CL_paramsT = {
    0:('D',100),
    1:('N', 0),
    2:('D',200),
    3:('N',0)
}

#%%
solver1 = Solver(mesh_objQ1,Lx,Ly,Gamma,q,rho,Cp,CL_paramsT,velocityField,'upwind')
solver1.solve(5)
solver1.plot("Solver 1 T")

#%%
solver2 = Solver(mesh_objQ2,Lx,Ly,0.5,1000,CL_paramsT)
solver2.solve()
solver2.plot()

#%%


solver1.defineSolAnalytique(SolAnalytique)
solver2.defineSolAnalytique(SolAnalytique)

#%%
E1 = solver1.errorQuadratique()
E2 = solver2.errorQuadratique()

print(E1,E2)

h1 = solver1.getMeanElementSize()
h2 = solver2.getMeanElementSize()

print(h1,h2)
#%%
OdC = np.log(E1/E2)/np.log(h1/h2)
print("Ordre de convergence : {}".format(OdC))

#%%
print("Solver 1")
solver1.coupeY(0.5,False)
solver1.coupeX(0.5,False)

#%%
print("Solver 2")
solver2.coupeY(0.5)

#%%
solver1._util.pause()