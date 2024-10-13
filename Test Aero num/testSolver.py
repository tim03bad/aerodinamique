#%%

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


from solver_original import Solver
from utilities import Utilities as util


from mesh import Mesh
from meshGenerator import MeshGenerator

from meshGenerator import MeshGenerator

#%%
T0 = 400
Tx = 50
Txy = 100

q=1000
Gamma = 0.1
rho = 1
Lx = 1
Ly = 1

#%%


x,y = sp.symbols('x y')

# T = T0 + Tx*sp.cos(sp.pi*x) + Txy*sp.sin(sp.pi*x*y)
u = 2.5
v = 0
velocityField = [u,v]

# fT,gradT,S = util.computeMMS(x,y,T,rho,k,Cp,velocityField)
fV = sp.lambdify((x,y),velocityField,'numpy')
# fS = sp.lambdify((x,y),S,'numpy')


def fT(x,y):
    return 1 + (1-np.exp(25*x))/7.2e10 + 0*y




#%%
CL = {
    0 : ('D',1),
    1 : ('N',0),
    2 : ('D',0),
    3 : ('N',0)

}


#%%
#lcs = [0.04,0.06,0.08,0.1,0.2,0.3,0.4,0.5]
lcs = [0.2]
bounds = [0,1,0,1]


Elist = []
Hlist = []

for lc in lcs:
    mesh, elements = util.prepareMesh({'mesh_type': 'QUAD', 'Nx': 100, 'Ny': 5},bounds,0)
    solver = Solver(mesh, elements,fV,CL,rho,Gamma,0,verbose=True,scheme='upwind')
    solver.solve()
    FieldN = solver._solution
    FieldA = util.getAnalyticalField(elements,fT)
    X,T = util.coupeY(mesh,FieldN,.5,False)
    Ta = fT(X,.5)
    plt.plot(X,T)
    plt.plot(X,Ta)
    plt.title(" T(x) (lc = {})".format(lc))
    plt.show()

    util.plotField(mesh,FieldN,"T",FieldA)

    Elist.append(util.quadraticError(FieldA,FieldN))
    Hlist.append(util.meanSize(elements))


#%%
E = np.array(Elist)
H = np.array(Hlist)
plt.plot(np.log(H),np.log(E))
plt.grid()
plt.xlabel("log(h)")
plt.ylabel("log(e)")
plt.title("Ordre de convergence (ln(E)=f(ln(h)))")
plt.show()
p = np.polyfit(np.log(H[0:3]),np.log(E[0:3]),1)
print("Pente = {}".format(p[0]))


#%%