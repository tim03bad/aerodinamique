#%%
from mesh import Mesh
from meshGenerator import MeshGenerator
from meshPlotter import MeshPlotter
from meshConnectivity import MeshConnectivity

import numpy as np
from element import Element
import sympy as sp
import pyvista as pv
import pyvistaqt as pvQt
import matplotlib.pyplot as plt
import leastSquareSolver as lss
from utilities import Utilities as util


plotter = MeshPlotter()

#%%
x,y = sp.symbols('x y')
T = sp.cos(sp.pi*x) + sp.sin(sp.pi*y)
T_x = T.diff(x)
T_y = T.diff(y)
Tgrad = [T_x, T_y]

fT = sp.lambdify((x,y), T, 'numpy')
gradT = sp.lambdify((x,y), Tgrad, 'numpy')

#%% CL

CL = {
    0: ('N',gradT),
    1: ('N',gradT),
    2: ('N',gradT),
    3: ('N',gradT)
}


#%%

def getFieldValue(mesh: Mesh,Elements: list[Element], gradT):

    GradN = np.zeros(mesh.get_number_of_elements())
    Grad = np.zeros((mesh.get_number_of_elements(), 2))
    for e in Elements:
        GradN[e._index] = np.linalg.norm(e._grad)
        Grad[e._index] = e._grad

    ValueAnalytique = np.zeros(mesh.get_number_of_elements())
    GradAnalytique = np.zeros((mesh.get_number_of_elements(), 2))
    for e in Elements:
        Center = e._center
        ValueAnalytique[e._index] = np.linalg.norm(gradT(Center[0], Center[1]))
        GradAnalytique[e._index] = gradT(Center[0], Center[1])

    return GradN, Grad, ValueAnalytique, GradAnalytique

def plotField(mesh: Mesh,fieldN,fieldA,title:str):

    pl = pvQt.BackgroundPlotter(shape=(1, 2))  # Allow the execution to continue

    nodes, elements = plotter.prepare_data_for_pyvista(mesh)


    pv_mesh = pv.PolyData(nodes, elements)
    pv_mesh[title] = fieldN

    pv_mesh2 = pv.PolyData(nodes, elements)
    pv_mesh2["Erreur"] = fieldA-fieldN


    pl.subplot(0, 0)
    pl.add_mesh(pv_mesh, show_edges=True, line_width=2, cmap="hot",scalars=title)
    pl.add_text("Norme numérique", position="upper_left")
    pl.subplot(0, 1)
    pl.add_mesh(pv_mesh2, show_edges=True, line_width=2, cmap="cool",scalars="Erreur")
    pl.add_text("Différence", position="upper_left")
    pl.show()


def errorQ(ValueAnalytique, GradN):

    D = (ValueAnalytique-GradN)**2
    D = np.mean(D)
    D = np.sqrt(D)
    return D

def norme2(Grad,GradA):
    D = GradA-Grad
    D = np.linalg.norm(D,axis=0)**2
    D = np.mean(D)
    return np.sqrt(D)
    

def meanSize(Elements: list[Element]):
    Area = np.zeros(mesh.get_number_of_elements())
    for e in Elements:
        Area[e._index] = e._area
    return np.mean(Area)**0.5

def sizeUniformMesh(mesh: Mesh):
    return mesh.get_number_of_elements()**0.5

#%%
Elist = []
Hlist = []

#lcs = [10,20,30,40,50,80,100,150] # pour quad
lcs = [0.03,0.05,0.07,0.09,0.1,0.2,0.3,0.4,0.5]
for lc in lcs:
    mesh, Elements = util.prepareMesh({'mesh_type': 'TRI', 'lc': lc},fT)
    
    lssSolver = lss.LeastSquareSolver(mesh,Elements,CL)
    lssSolver.solve()

    GradN, Grad, ValueAnalytique, GradAnalytique = getFieldValue(mesh,Elements,gradT)
    e = errorQ(ValueAnalytique, GradN)
    h = meanSize(Elements)
    Elist.append(e)
    Hlist.append(h)
    plotField(mesh,GradN,ValueAnalytique,"GradN {}".format(lc))


#%% 

#%%

#%%

#%%
Enumpy = np.array(Elist)
Hnumpy = np.array(Hlist)
indexes = np.argsort(Hnumpy)
Enumpy = Enumpy[indexes]
Hnumpy = Hnumpy[indexes]
print(Enumpy)
print(Hnumpy)

#%%
plt.plot(np.log(Hnumpy), np.log(Enumpy))
plt.grid()

p = np.polyfit(np.log(Hnumpy[0:5]), np.log(Enumpy[0:5]), 1)
print(p)