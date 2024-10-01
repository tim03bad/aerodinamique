#%% Import
from re import L
import numpy as np


from mesh import Mesh
from meshConnectivity import MeshConnectivity
from meshGenerator import MeshGenerator
from meshPlotter import MeshPlotter

from meanSquare import MeanSquare
from element import Element

from solver import Solver

#%% Geométrie du maillage et donnée

Lx = 1
Ly = 1


mesh_params_Q = {
    'mesh_type': 'QUAD',
    'Nx': 100,
    'Ny': 1
}

mesher = MeshGenerator(verbose=True)
mesh_objQ = mesher.rectangle([0, Lx, 0, Ly], mesh_params_Q)
conec = MeshConnectivity(mesh_objQ)
conec.compute_connectivity()


#%%
CL_paramsT = {
    0:('D',100),
    1:('N', 0),
    2:('D',200),
    3:('N',0)
}

#%%
solver = Solver(mesh_objQ,Lx,Ly,0.5,1000,CL_paramsT)


#%%
solver.solve()
solver.plot()

#%%
Gamma = 0.5
q = 1000
def SolAnalytique(x,y):
    return (100/Lx + (q/(2*Gamma))*(Lx-x))*x + 100 + 0*y
solver.defineSolAnalytique(SolAnalytique)
print(solver.errorQuadratique())