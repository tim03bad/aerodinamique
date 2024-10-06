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

mesh_params_Q1 = {
    'mesh_type': 'TRI',
    'lc': 0.08
}

mesh_params_Q2 = {
    'mesh_type': 'TRI',
    'lc': 0.02
}

mesher = MeshGenerator(verbose=True)
mesh_objQ1 = mesher.rectangle([0, Lx, 0, Ly], mesh_params_Q1)
conec = MeshConnectivity(mesh_objQ1)
conec.compute_connectivity()

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
solver1 = Solver(mesh_objQ1,Lx,Ly,0.5,1000,CL_paramsT)
solver1.solve()
#solver1.plot()

#%%
solver2 = Solver(mesh_objQ2,Lx,Ly,0.5,1000,CL_paramsT)
solver2.solve()
#solver2.plot()

#%%
Gamma = 0.5
q = 1000

def SolAnalytique(x,y):
    return (100/Lx + (q/(2*Gamma))*(Lx-x))*x + 100 + 0*y

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
solver1._util.pause()
X, phi = solver1.coupeY(0.5)

# Combiner les deux listes en une liste de tuples (abscisse, ordonnée)
couples = list(zip(X, phi))

# Trier la liste de tuples en fonction des valeurs d'abscisses (le premier élément de chaque tuple)
couples_tries = sorted(couples, key=lambda x: x[0])

# Séparer les abscisses et ordonnées triées
X_tries, phi_tries = zip(*couples_tries)

# Convertir en liste si nécessaire
abscisses_tries = list(X_tries)
ordonnees_tries = list(phi_tries)

plt.plot(abscisses_tries, ordonnees_tries)
plt.show()