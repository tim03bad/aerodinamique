# -*- coding: utf-8 -*-
"""
Programme d'utilisation du solveur diffusion thermoque
MEC6616 - Aérodynamique numérique
Date de création: 2022 - 01 - 04
Auteur 1: Emilien Vigier
Auteur 2: Timothée Badiche
"""

#Libraries et modules
from re import L, T
import numpy as np
import matplotlib.pyplot as plt

from mesh import Mesh
from meshConnectivity import MeshConnectivity
from meshGenerator import MeshGenerator
from meshPlotter import MeshPlotter

from meanSquare import MeanSquare
from element import Element

from solver import Solver
import solver
from utilities import Utilities

#Parametres
Lx = 0.02    #2cm
Ly = 0.1     #10cm

Ta = 100
Tb = 200

Gamma = 0.5   #Conduction thermique
q = 1000e3    #Source thermique (W/m^3)

CL_params = {
    0: ('D', Ta),
    1: ('N', 0),
    2: ('D', Tb),
    3: ('N', 0)
}

############# Question 1 ################

#Solution analytique
def SolAnalytique(x,y):
    return ((Tb-Ta)/Lx + (q/(2*Gamma))*(Lx-x))*x + Ta + 0*y


mesh_params_T = {
    'mesh_type': 'TRI',
    'lc': 0.001
}

mesh_params_Q = {
    'mesh_type': 'QUAD',
    'Nx': 25,
    'Ny': 5
}

mesher = MeshGenerator(verbose=True)

mesh_objQ = mesher.rectangle([0, Lx, 0, Ly], mesh_params_Q)
conecQ = MeshConnectivity(mesh_objQ)
conecQ.compute_connectivity()

mesh_objT = mesher.rectangle([0, Lx, 0, Ly], mesh_params_T)
conecT = MeshConnectivity(mesh_objT)
conecT.compute_connectivity()

solverQ = Solver(mesh_objQ, Lx, Ly, Gamma, q, CL_params)
solverQ.solve()
solverQ.defineSolAnalytique(SolAnalytique)
solverQ.plot()

solverT = Solver(mesh_objT, Lx, Ly, Gamma, q, CL_params)
solverT.solve()
solverT.defineSolAnalytique(SolAnalytique)
solverT.plot()

#Coupe 
Xq,Tq =solverQ.coupeY(0.05)
Xt,Tt = solverT.coupeY(0.05)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 8))

ax1.plot(Xq,Tq,'rx',label='Numerique')
ax1.plot(Xq,SolAnalytique(Xq,0.05),'b-',label='Analytique')
ax1.set_title('Quadrilatères transfinis')
ax1.set_ylabel('Temperature (K)')
ax1.set(xlim=(0, Lx), ylim=(0, 300))
ax1.legend()
ax1.grid()

ax2.plot(Xt,Tt,'bx',label='Numerique')
ax2.plot(Xt,SolAnalytique(Xt,0.05),'r-',label='Analytique')
ax2.set_title('Triangles non structurés')
ax2.set_xlabel('Position (m)')
ax2.set_ylabel('Temperature (K)')
ax2.set(ylim=(0, 300))
ax2.legend()
ax2.grid()

plt.tight_layout()
plt.show()

########### Erreur et convergence Quad ###########
Lx = 1
Ly = 1
q = 1000

mesh_params_Q1 = {
    'mesh_type': 'QUAD',
    'Nx': 50,
    'Ny': 5
}

mesh_params_Q2 = {
    'mesh_type': 'QUAD',
    'Nx': 100,
    'Ny': 5
}

mesher = MeshGenerator(verbose=False)

mesh_objQ1 = mesher.rectangle([0, Lx, 0, Ly], mesh_params_Q1)
conecQ1 = MeshConnectivity(mesh_objQ1)
conecQ1.compute_connectivity()

mesh_objQ2 = mesher.rectangle([0, Lx, 0, Ly], mesh_params_Q2)
conecQ2 = MeshConnectivity(mesh_objQ2)
conecQ2.compute_connectivity()

solverQ1 = Solver(mesh_objQ1, Lx, Ly, Gamma, q, CL_params)
solverQ1.solve()
solverQ1.defineSolAnalytique(SolAnalytique)
solverQ1.plot()

solverQ2 = Solver(mesh_objQ2, Lx, Ly, Gamma, q, CL_params)
solverQ2.solve()
solverQ2.defineSolAnalytique(SolAnalytique)
solverQ2.plot()

#Erreur (L2)
E1 = solverQ1.errorQuadratique()
E2 = solverQ2.errorQuadratique()

#Taille moyenne des éléments
h1 = solverQ1.getMeanElementSize()
h2 = solverQ2.getMeanElementSize()

OdC = np.log(E1/E2)/np.log(h1/h2)
print("Ordre de convergence Quad Transfini : {}".format(OdC))

Utilities.pause()

########### Erreur et convergence Tri ###########
Lx = 1
Ly = 1
q = 1000

mesh_params_T1 = {
    'mesh_type': 'TRI',
    'lc': 0.1
}

mesh_params_T2 = {
    'mesh_type': 'TRI',
    'lc': 0.05
}

mesher = MeshGenerator(verbose=False)

mesh_objT1 = mesher.rectangle([0, Lx, 0, Ly], mesh_params_T1)
conecT1 = MeshConnectivity(mesh_objT1)
conecT1.compute_connectivity()

mesh_objT2 = mesher.rectangle([0, Lx, 0, Ly], mesh_params_T2)
conecT2 = MeshConnectivity(mesh_objT2)
conecT2.compute_connectivity()

solverT1 = Solver(mesh_objT1, Lx, Ly, Gamma, q, CL_params)
solverT1.solve()
solverT1.defineSolAnalytique(SolAnalytique)
solverT1.plot()

solverT2 = Solver(mesh_objT2, Lx, Ly, Gamma, q, CL_params)
solverT2.solve()
solverT2.defineSolAnalytique(SolAnalytique)
solverT2.plot()

#Erreur (L2)
E1 = solverT1.errorQuadratique()
E2 = solverT2.errorQuadratique()
print("E1 = {}, E2 = {}".format(E1,E2))

#Taille moyenne des éléments

h1 = solverT1.getMeanElementSize()
h2 = solverT2.getMeanElementSize()
print("h1 = {}, h2 = {}".format(h1,h2))

OdC = np.log(E1/E2)/np.log(h1/h2)
print("Ordre de convergence Triangle : {}".format(OdC))

Utilities.pause()


########### Avec/Sans cross diffusion ###########
Lx = 1
Ly = 1
q = 1000

mesh_params_T = {
    'mesh_type': 'TRI',
    'lc': 0.08
}

mesh_objT1 = mesher.rectangle([0, Lx, 0, Ly], mesh_params_T)
conecT1 = MeshConnectivity(mesh_objT1)
conecT1.compute_connectivity()

solverT1 = Solver(mesh_objT1, Lx, Ly, Gamma, q, CL_params)
solverT1.solve()

mesh_objT2 = mesher.rectangle([0, Lx, 0, Ly], mesh_params_T)
conecT2 = MeshConnectivity(mesh_objT2)
conecT2.compute_connectivity()

solverT2 = Solver(mesh_objT2, Lx, Ly, Gamma, q, CL_params, crossDiffusion=False)
solverT2.solve()

#Coupe 
X1,T1 =solverT1.coupeY(0.5)
X2,T2 = solverT2.coupeY(0.5)

print("Les deux solutions sont extrement proches en raison de la structure en triangle isocèle d'une bonne partie\n du maillage, de ce fait Pksieta est souvant nul")


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 8))

ax1.plot(X1,T1,'rx',label='Numerique')
ax1.plot(X1,SolAnalytique(X1,0.5),'b-',label='Analytique')
ax1.set_title('Avec cross diffusion')
ax1.set_ylabel('Temperature (K)')
ax1.set(xlim=(0, Lx), ylim=(0, 450))
ax1.legend()
ax1.grid()

ax2.plot(X2,T2,'bx',label='Numerique')
ax2.plot(X2,SolAnalytique(X2,0.5),'r-',label='Analytique')
ax2.set_title('Sans cross diffusion')
ax2.set_xlabel('Position (m)')
ax2.set_ylabel('Temperature (K)')
ax2.set(ylim=(0, 450))
ax2.legend()
ax2.grid()

plt.tight_layout()
plt.show()




