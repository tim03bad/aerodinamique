"""
Programme LAP6 : Momentum2DSolver.py
Author 1 : Emilien Vigier
Author 2 : Timothée Badiche
Date : 11/11/2024
"""

#%%Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import pyvista as pv
import pyvistaqt as pvQt

#Import Classes
import meshPlotter
from momentum2DSolver import Momentum2DSolver
from utilities import Utilities as util
from element import Element

#Import Mesh modules
from mesh import Mesh
from meshGenerator import MeshGenerator
from meshPlotter import MeshPlotter

#%%
#Paremeters definition
b=1
mu=0.1
rho=1

#Condition au limite
x,y = sp.symbols('x,y')
u = 10-x + 0*y
fU = sp.lambdify((x,y),u,'numpy')

#%%
CLx = {
    0 : ('D',fU,'E'),
    1 : ('D',fU,'W'),
    2 : ('D',fU,'S'),
    3 : ('D',fU,'W')
}

Cly = {
    0 : ('D',0),
    1 : ('D',0),
    2 : ('D',0),
    3 : ('D',0)
}

#%%
P = 0
#Champs de pression
fP = lambda x,y : 0*x + 0*y





#%% Cas A 9 quadrilatères
print("## --> Cas A : 9 quadrilatères")

mesh, elements = util.prepareMesh({'mesh_type': 'QUAD', 'Nx': 3, 'Ny': 3},[0,10,0,10],0)
solver = Momentum2DSolver(mesh,elements,CLx,Cly,rho,mu,fP)
res = solver.solve(Rmax=0.01)

util().plotVectorField(mesh,res,"Champs de vitesse QUAD P={}".format(P),"Velocity")

print("""Observation : On observe que correction de vitesse annule bien la divergence du champs de vitesse ,
       ce qui signifie que l'on a bien bien un champs de vitesse correspondant à un écoulement incompressible. 
      
      """)
util().pause()


#%%Meshing and solving
print("## --> Cas B : 18 triangles")
mesh, elements = util.prepareMesh({'mesh_type': 'TRI', 'Nx': 3, 'Ny': 3},[0,10,0,10],0)
solver = Momentum2DSolver(mesh,elements,CLx,Cly,rho,mu,fP)
res = solver.solve(Rmax=0.01)

util().plotVectorField(mesh,res,"18 Triangles P={}".format(P),"Velocity")

print("""Observation : On observe que correction de vitesse annule bien la divergence du champs de vitesse ,
       ce qui signifie que l'on a bien bien un champs de vitesse correspondant à un écoulement incompressible. 
      
      """)

util().pause()



#%%



