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

#%% Calcul de l'ordre de convergence 

def calcul_convergence(solver_grossier, solver_fin, schema, cond_limite):
            
    solver_grossier.defineSolAnalytique(fT)
    solver_fin.defineSolAnalytique(fT)
    E1 = solver_grossier.errorQuadratique()
    E2 = solver_fin.errorQuadratique()
    
    OdC = np.log(E1/E2)/np.log(calculTailleMoyenne_grossier/calculTailleMoyenne_fin)
    print("Ordre de convergence de schema {} et de condition limite {} : {}".format(schema, cond_limite, OdC))
        
#%% Plot de la coupe avec comparaison avec la solution analytique

def coupe_comparaison(solver, plan, coord):
    if plan=='X':
        D, T_num = solver.coupeX(coord, False)
        T_ana = [fT(coord,y) for y in D]
    if plan=='Y':
        D, T_num = solver.coupeX(coord, False)
        T_ana = [fT(x,coord) for x in D]
    
    plt.plot(D, T_num, label="solution numérique")
    plt.plot(D, T_ana, label="solution numérique")
    plt.xlabel("Distance en m")
    plt.ylabel("Température en K")
    plt.show()

#%% Calcul la dimension moyenne d'un élément

def mesh_constant(mesh_obj):
    listOfElement = [Element(mesh_obj, i) for i in range(mesh_obj.get_number_of_elements())]
    listOfArea = np.array([E.get_Area() for E in listOfElement])
    calculTailleMoyenne = np.sqrt(np.sum(listOfArea)/len(listOfArea))
    
    listOfVelocity = [velocityField(e.get_Coord()[0], e.get_Coord()[1]) for e in listOfElement]
    listOfNorm = np.array([np.sqrt(v[0]**2 + v[1]**2) for v in listOfVelocity])
    calculVitesseMoyenne = np.sum(listOfNorm)/len(listOfNorm)

    return calculTailleMoyenne, calculVitesseMoyenne

#%% Creation des mesh

#mesh pour le cas avec maillage fin
mesh_params_fin = {
    'mesh_type': 'TRI',
    'lc': 0.05
}

mesher_fin = MeshGenerator(verbose=True)
mesh_obj_fin = mesher_fin.rectangle([-1, 1, -1, 1], mesh_params_fin)
conec_fin = MeshConnectivity(mesh_obj_fin)
conec_fin.compute_connectivity()


#mesh pour le cas avec maillage grossier
mesh_params_grossier = {
    'mesh_type': 'TRI',
    'lc': 0.2
}

mesher_grossier = MeshGenerator(verbose=True)
mesh_obj_grossier = mesher_grossier.rectangle([-1, 1, -1, 1], mesh_params_grossier)
conec_grossier = MeshConnectivity(mesh_obj_grossier)
conec_grossier.compute_connectivity()

calculTailleMoyenne_fin, calculVitesseMoyenne_fin = mesh_constant(mesh_obj_fin)
calculTailleMoyenne_grossier, calculVitesseMoyenne_grossier = mesh_constant(mesh_obj_grossier)

#%%
fT = sp.lambdify((x, y), Tmms, 'numpy')

#%% Calcul la solution numérique en fonction du nombre de Peclet choisi, du schema et des condtions limites 

Utilities.plot(mesh_obj_grossier,fT,"T analytique grossier",True,True)
Utilities.plot(mesh_obj_fin,fT,"T analytique fin",True,True)
choix = input("Choisissez Peclet parmi 1(1), 100(2) ou 10000(3) : ")
Pes = [1, 100, 10000]
if int(choix) in [1, 2, 3]:
    Pe = Pes[int(choix)-1]
    
    k_fin = calculTailleMoyenne_fin*calculVitesseMoyenne_fin*rho*Cp/Pe
    k_grossier = calculTailleMoyenne_grossier*calculVitesseMoyenne_grossier*rho*Cp/Pe
    
    #Terme source de correction 
    S_fin = rho*Cp*(uTmms_x + vTmms_y) - k_fin*(Tmms_xx + Tmms_yy)
    S_grossier = rho*Cp*(uTmms_x + vTmms_y) - k_grossier*(Tmms_xx + Tmms_yy)
    
    #Transformation en fonction numpy
    fS_fin = sp.lambdify((x, y), S_fin, 'numpy')
    
    fS_grossier = sp.lambdify((x, y), S_grossier, 'numpy')
    
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
        
        # Solver
        choix = input("Choisissez le schema parmi central(1) ou upwind(2) : ")
        schemas = ['central', 'upwind']
        if int(choix) in [1, 2]:
            schema = schemas[int(choix)-1]
            solver_fin = Solver(mesh_obj_fin,2,2,k_fin,fS_fin,rho,Cp,CL_params,velocityField,schema)
            solver_grossier = Solver(mesh_obj_grossier,2,2,k_grossier,fS_grossier,rho,Cp,CL_params,velocityField,schema)
            
            solver_fin.solve(4)
            solver_grossier.solve(4)
            Value_fin = solver_fin.plot("T fin {} \n Peclet = {}, Schema = {}".format(cond, Pe, schema),True,True)
            Value_grossier = solver_grossier.plot("T grossier {} \n Peclet = {}, Schema = {}".format(cond, Pe, schema),True,True)
            
            choix = input("Choisissez la coupe parmi X(1) ou Y(2) : ")
            coupes = ['X', 'Y']
            
            if int(choix) in [1, 2]:
                coupe = coupes[int(choix)-1]
                choix = input("Choisissez la coordonnée de la coupe en {} comprise entre -1 et 1 : ".format(coupe))
                coord = float(choix)
                
                if coord<1 and coord>=-1:
                    
                    calcul_convergence(solver_grossier, solver_fin, schema, cond)
                    coupe_comparaison(solver_grossier, coupe, coord)
                    coupe_comparaison(solver_fin, coupe, coord)
                    
                    Utilities.plotError(mesh_obj_grossier,fT,Value_grossier,"Erreur T-T2 grossier {} \n Schema = {}".format(cond, schema),True)
                    Utilities.plotError(mesh_obj_fin,fT,Value_fin,"Erreur T-T2 fin {} \n Schema = {}".format(cond, schema),True)
                    
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
    
