#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import pyvista as pv
import pyvistaqt as pvQt

#Import Classes
from momentum2DSolver import Momentum2DSolver
from utilities import Utilities as util
from element import Element

#Import Mesh modules
from mesh import Mesh
from meshGenerator import MeshGenerator
from meshPlotter import MeshPlotter

#Paremeters definition
b=1
U=1
mu=1
rho=1

#Condition au limite
CLx = {
    0 : ('N',0),
    1 : ('D',0),
    2 : ('N',0),
    3 : ('D',U)
}

Cly = {
    0 : ('N',0),
    1 : ('D',0),
    2 : ('N',0),
    3 : ('D',0)
}

print("############# Test Rhie-Crow sur maillage Triangle grossier lc = 0.2 #############")
print("--> P = 0 : gradient nul")
P = 0
#Champs de pression
fP = lambda x,y : 0*x + 0*y
#Solution analytique du champ de vitesse u
uA = lambda y : y*(1+P*(1-y))

lc = 0.2

#Meshing and solving
mesh, elements = util.prepareMesh({'mesh_type': 'TRI', 'lc': lc},[0,1,0,1],0)
solver = Momentum2DSolver(mesh,elements,CLx,Cly,rho,mu,fP)
res = solver.solve(Rmax=0.01)

util().plotVectorField(mesh,res,"Champs de vitesse TRI P={}".format(P),"Velocity")
#Coupe X = 0.5, champs de vitesse u et v
Ufx = res[:,0]
Ufy = res[:,1]
Yx,Ux = util.coupeX(mesh,Ufx,.5,False)
Yy,Uy = util.coupeX(mesh,Ufy,.5,False)

plt.ion()
plt.plot(Ux/U,Yx/b,label='u')
plt.plot(Uy/U,Yy/b,label='v')
plt.plot(uA(Yx)/U,Yx/b,'--',label=r'$u_{analytique}$')
plt.xlabel("Vitesse normalisée sur U")
plt.ylabel("Y/b")    
plt.legend()
plt.title(r"Profil de vitesse ($P = {}$)".format(P))    
plt.grid()
plt.show()

#Test Rheie-Crow
solver.testRhieCrow()

print("""\n On observe que dans le cas limite d'un gradient nulle, la méthode de Rhie-Crow correspond exactement au calcul de la vitesse normale à la face
      par moyenne simple des vitesses des élements gauche et droite\n""")

util().pause()
plt.close()


print("\n--> P = 3 : gradient constant")
P = 3
#Champs de pression (CL Pres = 0 en x = 1)
fP = lambda x,y : -2*P*x + 2*P + 0*y
#Solution analytique du champ de vitesse u
uA = lambda y : y*(1+P*(1-y))

lc = 0.2

#Meshing and solving
mesh, elements = util.prepareMesh({'mesh_type': 'TRI', 'lc': lc},[0,1,0,1],0)
solver = Momentum2DSolver(mesh,elements,CLx,Cly,rho,mu,fP)
res = solver.solve(Rmax=0.01)

util().plotVectorField(mesh,res,"Champs de vitesse TRI P={}".format(P),"Velocity")
#Coupe X = 0.5, champs de vitesse u et v
Ufx = res[:,0]
Ufy = res[:,1]
Yx,Ux = util.coupeX(mesh,Ufx,.5,False)
Yy,Uy = util.coupeX(mesh,Ufy,.5,False)

plt.plot(Ux/U,Yx/b,label='u')
plt.plot(Uy/U,Yy/b,label='v')
plt.plot(uA(Yx)/U,Yx/b,'--',label=r'$u_{analytique}$')
plt.xlabel("Vitesse normalisée sur U")
plt.ylabel("Y/b")    
plt.legend()
plt.title(r"Profil de vitesse ($P = {}$)".format(P))    
plt.grid()
plt.show()

#Test Rheie-Crow
solver.testRhieCrow()

print("""\n On observe que dans le cas limite d'un gradient nulle, la méthode de Rhie-Crow correspond exactement au calcul de la vitesse normale à la face
      par moyenne simple des vitesses des élements gauche et droite""")

util().pause()
plt.close()

## Maillage Quad#########################################################################
print("\n############# Test Rhie-Crow sur maillage Quad 8x8 ############")
print("--> P = 0 : gradient nul")
P = 0
#Champs de pression
fP = lambda x,y : 0*x + 0*y
#Solution analytique du champ de vitesse u
uA = lambda y : y*(1+P*(1-y))

N = 8

#Meshing and solving
mesh, elements = util.prepareMesh({'mesh_type': 'QUAD', 'Nx': N, 'Ny': N},[0,1,0,1],0)
solver = Momentum2DSolver(mesh,elements,CLx,Cly,rho,mu,fP)
res = solver.solve(Rmax=0.01)

util().plotVectorField(mesh,res,"Champs de vitesse QUAD P={}".format(P),"Velocity")

#Coupe X = 0.5, champs de vitesse u et v
Ufx = res[:,0]
Ufy = res[:,1]
Yx,Ux = util.coupeX(mesh,Ufx,.5,False)
Yy,Uy = util.coupeX(mesh,Ufy,.5,False)

plt.plot(Ux/U,Yx/b,label='u')
plt.plot(Uy/U,Yy/b,label='v')
plt.plot(uA(Yx)/U,Yx/b,'--',label=r'$u_{analytique}$')
plt.xlabel("Vitesse normalisée sur U")
plt.ylabel("Y/b")    
plt.legend()
plt.title(r"Profil de vitesse ($P = {}$)".format(P))    
plt.grid()
plt.show()

#Test Rheie-Crow
solver.testRhieCrow()

print("""\n On observe que dans le cas limite d'un gradient nulle, la méthode de Rhie-Crow correspond exactement au calcul de la vitesse normale à la face
      par moyenne simple des vitesses des élements gauche et droite\n""")

util().pause()
plt.close()

print("\n--> P = 3 : gradient constant")
P = 3
#Champs de pression (CL Pres = 0 en x = 1)
fP = lambda x,y : -2*P*x + 2*P + 0*y
#Solution analytique du champ de vitesse u
uA = lambda y : y*(1+P*(1-y))

N = 8

#Meshing and solving
mesh, elements = util.prepareMesh({'mesh_type': 'TRI', 'Nx': N, 'Ny': N},[0,1,0,1],0)
solver = Momentum2DSolver(mesh,elements,CLx,Cly,rho,mu,fP)
res = solver.solve(Rmax=0.01)

util().plotVectorField(mesh,res,"Champs de vitesse QUAD P={}".format(P),"Velocity")
#Coupe X = 0.5, champs de vitesse u et v
Ufx = res[:,0]
Ufy = res[:,1]
Yx,Ux = util.coupeX(mesh,Ufx,.5,False)
Yy,Uy = util.coupeX(mesh,Ufy,.5,False)

plt.plot(Ux/U,Yx/b,label='u')
plt.plot(Uy/U,Yy/b,label='v')
plt.plot(uA(Yx)/U,Yx/b,'--',label=r'$u_{analytique}$')
plt.xlabel("Vitesse normalisée sur U")
plt.ylabel("Y/b")    
plt.legend()
plt.title(r"Profil de vitesse ($P = {}$)".format(P))    
plt.grid()
plt.show()

#Test Rheie-Crow
solver.testRhieCrow()

print("""\n On observe que dans le cas limite d'un gradient nulle, la méthode de Rhie-Crow correspond exactement au calcul de la vitesse normale à la face
      par moyenne simple des vitesses des élements gauche et droite""")

util().pause()
plt.close()




