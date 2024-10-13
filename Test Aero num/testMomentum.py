#%%

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


from momentum2DSolver import Momentum2DSolver
from utilities import Utilities as util


from mesh import Mesh
from meshGenerator import MeshGenerator
from meshPlotter import MeshPlotter
from meshGenerator import MeshGenerator

#%%
b = 1
U = 1
mu = 1
rho = 1

P = 1 
gradP = [-2*P,0]

#%%
def uA(y):
    return U*(y/b) + gradP[0]*(y**2-b*y)/(2*mu)

#%% Mesh preping
mesh,elements = util.prepareMesh({'mesh_type': 'TRI', 'lc': 0.1},[0,1,0,1],0)
plotter = MeshPlotter()
#plotter.plot_mesh(mesh,label_faces=True)

#%% Conditions aux limites
CL = {
    0 : ('N',0),
    1 : ('D',0),
    2 : ('N',0),
    3 : ('D',U)
}

#%%
solver = Momentum2DSolver(mesh,elements,CL,rho,mu,gradP,scheme='central')

#%%

res = solver.solve(Rmax=0.001)

#%%
Uf = res[:,0]
Y,Uy = util.coupeX(mesh,Uf,.5,False)

#%%
plt.plot(Uy/U,Y/b)
plt.plot(uA(Y)/U,Y/b)
plt.show()

