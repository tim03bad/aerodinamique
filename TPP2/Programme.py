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
import solver




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
    'lc': 0.08
}

mesher = MeshGenerator(verbose=True)
mesh_obj = mesher.rectangle([-1, 1, -1, 1], mesh_params)
conec = MeshConnectivity(mesh_obj)
conec.compute_connectivity()

CL_params = {
    0:('D',fT),
    1:('D',fT),
    2:('D',fT),
    3:('D',fT),
}

#%% Solver
solver = Solver(mesh_obj,2,2,k,fS,rho,Cp,CL_params,velocityField,'upwind')
solver.solve()
Value = solver.plot()

# %%
import pyvista as pv
import pyvistaqt as pvQt
from meshPlotter import MeshPlotter

plotter = MeshPlotter()
nodes,elements = plotter.prepare_data_for_pyvista(mesh_obj)
pv_mesh = pv.PolyData(nodes, elements)

cell_centers = np.zeros((mesh_obj.get_number_of_elements(), 2))
for i_element in range(mesh_obj.get_number_of_elements()):
    center_coords = np.array([0.0, 0.0])
    nodes = mesh_obj.get_element_to_nodes(i_element)
    for node in nodes:
        center_coords[0] += mesh_obj.get_node_to_xcoord(node)
        center_coords[1] += mesh_obj.get_node_to_ycoord(node)
    center_coords /= nodes.shape[0]
    cell_centers[i_element, :] = center_coords



pv_mesh['Champ T'] = fT(cell_centers[:, 0], cell_centers[:, 1])-Value

pl = pvQt.BackgroundPlotter()
pl.add_mesh(pv_mesh, scalars='Champ T', show_edges=True, cmap='hot')

pl.camera_position = 'xy'
pl.show_grid()
pl.show()