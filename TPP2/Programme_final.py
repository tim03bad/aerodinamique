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


from meshConnectivity import MeshConnectivity
from meshGenerator import MeshGenerator

from solver import Solver
from utilities import Utilities
from element import Element



#%%Parametre physique
rho = 1
Cp = 1

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

def fct(x, y):
    return T0 + Tx*np.cos(np.pi*x) + Txy*np.sin(np.pi*x*y)

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

#%% Meshing
mesh_params = {
    'mesh_type': 'TRI',
    'lc': 0.2
}

mesher = MeshGenerator(verbose=True)
mesh_obj = mesher.rectangle([-1, 1, -1, 1], mesh_params)
conec = MeshConnectivity(mesh_obj)
conec.compute_connectivity()

mesh_params_fin = {
    'mesh_type': 'TRI',
    'lc': 0.05
}

mesher_fin = MeshGenerator(verbose=True)
mesh_obj_fin = mesher.rectangle([-1, 1, -1, 1], mesh_params_fin)
conec_fin = MeshConnectivity(mesh_obj_fin)
conec_fin.compute_connectivity()

#%% Calcul de la vitesse moyenne et de la dimension moyenne d'un élément
def mesh_constant(mesh_obj):
    listOfElement = [Element(mesh_obj, i) for i in range(mesh_obj.get_number_of_elements())]
    listOfArea = np.array([E.get_Area() for E in listOfElement])
    calculTailleMoyenne = np.sqrt(np.sum(listOfArea)/len(listOfArea))
    
    listOfVelocity = [velocityField(elem.get_Coord()[0], elem.get_Coord()[1]) for elem in listOfElement]
    VelocityNorm = [np.sqrt(v[0]**2 + v[1]**2) for v in listOfVelocity]
    calculVitesseMoyenne =  sum(VelocityNorm)/len(VelocityNorm)
    return calculTailleMoyenne, calculVitesseMoyenne

#%%
fT = sp.lambdify((x, y), Tmms, 'numpy')
Utilities.plot(mesh_obj,fT,"T analytique",True,True)

#%% Boucle pour chaque valeurs de Peclet
choix = input("Choisissez Peclet parmi 1(1), 100(2) ou 10000(3) : ")
Pes = [1, 100, 10000]
if int(choix) in [1, 2, 3]:
    Pe = Pes[int(choix)-1]
    calculTailleMoyenne, calculVitesseMoyenne = mesh_constant(mesh_obj)
    calculTailleMoyenne_fin, calculVitesseMoyenne_fin = mesh_constant(mesh_obj_fin)
    
    k = calculTailleMoyenne*calculVitesseMoyenne*rho*Cp/Pe
    k_fin = k
    #Terme source de correction 
    S = rho*Cp*(uTmms_x + vTmms_y) - k*(Tmms_xx + Tmms_yy)
    S_fin = rho*Cp*(uTmms_x + vTmms_y) - k_fin*(Tmms_xx + Tmms_yy)
    
    #Transformation en fonction numpy
    fS = sp.lambdify((x, y), S, 'numpy')
    fS_fin = sp.lambdify((x, y), S_fin, 'numpy')
    
    choix = input("Choisissez les conditions limites parmi Dirichlet(1) ou Dirichlet et Neumann(2) : ")
    Cond_lim = ['Dirichlet', 'Dirichlet et Neumann']
    if int(choix) in [1, 2]:
        cond = Cond_lim[int(choix)-1]
        if cond=='Dirichlet':
            CL_params = { #Pure Dirichlet 
                0:('D',fT),
                1:('D',fT),
                2:('D',fT),
                3:('D',fT),
                
            }
        
        if cond=='Dirichlet et Neumann':
            CL_params = { #Mix Dirichlet et Neumann
                0:('D',fT),
                1:('N',fTGrad),
                2:('D',fT),
                3:('N',fTGrad),
            }
        
        #%% Solver
        choix = input("Choisissez le schema parmi central(1) ou upwind(2) : ")
        schemas = ['central', 'upwind']
        if int(choix) in [1, 2]:
            schema = schemas[int(choix)-1]
            solver = Solver(mesh_obj,2,2,k,fS,rho,Cp,CL_params,velocityField,schema)
            solver_fin = Solver(mesh_obj_fin,2,2,k_fin,fS_fin,rho,Cp,CL_params,velocityField,schema)
            
            solver.solve()
            solver_fin.solve()
            Value = solver.plot("T {} \n Peclet = {}, Schema = {}".format(cond, Pe, schema))
            
            choix = input("Choisissez la coupe parmi X(1) ou Y(2) : ")
            coupes = ['X', 'Y']
            
            if int(choix) in [1, 2]:
                coupe = coupes[int(choix)-1]
                choix = input("Choisissez la coordonnée de la coupe en {} comprise entre -1 et 1 : ".format(coupe))
                
                if float(choix)<1 and float(choix)>=-1:
                    if coupe=='X': 
                        X,Y = solver.coupeX(float(choix))
                        plt.xlabel("axe x (en m)")
                    else: 
                        X,Y = solver.coupeY(float(choix))
                        plt.xlabel("axe y (en m)")
                    plt.ylabel("température (en K)")
                    plt.plot(X,Y)
                    plt.show()
                    
                    solver.defineSolAnalytique(fct)
                    solver_fin.defineSolAnalytique(fct)
                    E1 = solver.errorQuadratique()
                    E2 = solver_fin.errorQuadratique()

                    OdC = np.log(E1/E2)/np.log(calculTailleMoyenne/calculTailleMoyenne_fin)
                    print("Ordre de convergence : {}".format(OdC))
                    
                    Utilities.plotError(mesh_obj,fT,Value,"Erreur T-T2 {} \n Peclet = {}, Schema = {}".format(cond, Pe, schema),True)
                    
                
                    # %%
                    Utilities.pause()
                else:
                    print("la coordonnée est hors des limites")
            else:
                print("La coupe n'est pas correcte")
        else:
            print("Le schema n'est pas correct")
    else:
        print("Les conditions limites ne sont pas correctes")
else:
    print("La valeur de Peclet n'est pas correcte")