# -*- coding: utf-8 -*-
"""
Programme d'utilisation du solveur diffusion thermoque
MEC6616 - Aérodynamique numérique
Date de création: 2022 - 01 - 04
Auteur 1: Emilien Vigier
Auteur 2: Timothée Badiche
"""

#%%Libraries et modules
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

from mesh import Mesh

from meshConnectivity import MeshConnectivity
from meshGenerator import MeshGenerator

from meshPlotter import MeshPlotter
from solver import Solver
from utilities import Utilities


import pyvista as pv
import pyvistaqt as pvQt

#%%Parametre physique
rho = 1
Cp = 1
k = 1

#%% MMS Sympy
####### Test MMS #######
# Construction de la solution MMS
x, y = sp.symbols('x y')

#Champ de vitesse
u = (2*x**2 - x**4 -1)*(y-y**3)
v = -(2*y**2 - y**4 -1)*(x-x**3)

fu = sp.lambdify((x, y), u, 'numpy')
fv = sp.lambdify((x, y), v, 'numpy')

def velocityField(x,y):
    return np.array([fu(x,y),fv(x,y)])


T0 = 400
Tx = 50
Txy = 100
Tmms = T0 + Tx*sp.cos(sp.pi*x) + Txy*sp.sin(sp.pi*x*y)

Tmms_x = sp.diff(Tmms, x)
Tmms_y = sp.diff(Tmms, y)
fTx = sp.lambdify((x, y), Tmms_x, 'numpy')
fTy = sp.lambdify((x, y), Tmms_y, 'numpy')

def fTGrad(x,y):
    return np.array([fTx(x,y),fTy(x,y)])

#Derivées
uTmms_x = sp.diff(Tmms*u, x)
vTmms_y = sp.diff(Tmms*v, y)

Tmms_xx = sp.diff(Tmms, x, 2)
Tmms_yy = sp.diff(Tmms, y, 2)

#Terme source de correction 
S = rho*Cp*(uTmms_x + vTmms_y) - k*(Tmms_xx + Tmms_yy)

#Transformation en fonction numpy
fT = sp.lambdify((x, y), Tmms, 'numpy')
fS = sp.lambdify((x, y), S, 'numpy')


#%% Meshing
mesh_params = {
    'mesh_type': 'TRI',
    'lc': 0.1
}

mesher = MeshGenerator(verbose=True)
mesh_obj = mesher.rectangle([-1, 1, -1, 1], mesh_params)
conec = MeshConnectivity(mesh_obj)
conec.compute_connectivity()

CL_params_1 = { #Pure Dirichlet 
    0:('D',fT),
    1:('D',fT),
    2:('D',fT),
    3:('D',fT),
    
}

CL_params_2 = { #Mix Dirichlet et Neumann
    0:('D',fT),
    1:('N',fTGrad),
    2:('D',fT),
    3:('N',fTGrad),
}

#%% Solver

#Dirichlet
solver1 = Solver(mesh_obj,2,2,k,fS,rho,Cp,CL_params_1,velocityField,'upwind')
solver1.solve()
Value1 = solver1.plot("T Dirichlet")


#Neumann+Dirichlet
solver2 = Solver(mesh_obj,2,2,k,fS,rho,Cp,CL_params_2,velocityField,'upwind')
solver2.solve()
Value2 = solver2.plot("T Neumann+Dirichlet")


#%%
Utilities.plot(mesh_obj,fT,"T analytique",True,True)
#%%
Utilities.plotError(mesh_obj,fT,Value2,"Erreur T-T2",True)


# %%
Utilities.pause()