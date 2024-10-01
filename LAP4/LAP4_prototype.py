#%% Import



import numpy as np
from regex import F

from mesh import Mesh
from meshConnectivity import MeshConnectivity
from meshGenerator import MeshGenerator
from meshPlotter import MeshPlotter

from meanSquare import MeanSquare
from element import Element

import pyvista as pv
import pyvistaqt as pvQt

#internal class

class dataFace:
    def __init__(self,Xa,Xb,Ya,Yb,XA,XP,YA,YP):
        """
        Constructeur de l'objet dataFace

        Parameters
        ----------
        Xa,Ya : float
            Coordonn es du premier point de la face
        Xb,Yb : float
            Coordonn es du deuxi me point de la face
        XA,YA : float
            Coordonn es du centre de l' l ment
        XP,YP : float
            Coordonn es du centre de l' l ment adjacent

        Sert de conteneur pour les coordonnées des points utile au calcul
        """
        self.Xa = Xa
        self.Xb = Xb
        self.Ya = Ya
        self.Yb = Yb
        self.XA = XA
        self.XP = XP
        self.YA = YA
        self.YP = YP

    
    def __str__(self):
        """
        Affichage des donn es de l'objet
        """
        return "Xa : {:.3f}, Xb : {:.3f}, Ya : {:.3f}, Yb : {:.3f}\nXA : {:.3f}, XP : {:.3f}, YA : {:.3f}, YP : {:.3f}".format(self.Xa,self.Xb,self.Ya,self.Yb,self.XA,self.XP,self.YA,self.YP)
#%% Fonctions

def FaceCoordinatesCalcultion(mesh_obj: Mesh,face : int):
    FaceNodes = mesh_obj.get_face_to_nodes(face)
    Elems = mesh_obj.get_face_to_elements(face)
    Eg = QuadList[Elems[0]]
    Ed = QuadList[Elems[1]]
    
    Xa = mesh_obj.node_to_xcoord[FaceNodes[0]]
    Xb = mesh_obj.node_to_xcoord[FaceNodes[1]]

    Ya = mesh_obj.node_to_ycoord[FaceNodes[0]]
    Yb = mesh_obj.node_to_ycoord[FaceNodes[1]]

    XP = Eg.ElementCoord[0]
    XA = Ed.ElementCoord[0]

    YP = Eg.ElementCoord[1]
    YA = Ed.ElementCoord[1]

    return dataFace(Xa,Xb,Ya,Yb,XA,XP,YA,YP)


def Pnksi(data : dataFace,DAi :float,dKsi : float):

    
    dXi = data.Xb - data.Xa
    dYi = data.Yb - data.Ya

    a = dYi*(data.XA-data.XP)/(DAi*dKsi)
    b = dXi*(data.YA-data.YP)/(DAi*dKsi)

    return a-b

def Pksieta(data : dataFace,dKsi,DAi):

    a = (data.XA-data.XP)*(data.Xa-data.Xb)/(dKsi*DAi)
    b = (data.YA-data.YP)*(data.Ya-data.Yb)/(dKsi*DAi)

    return a+b 

def Di(P_nksi : float,Gamma : float,DAi : float,dKsi : float):

    return (1/P_nksi)*(Gamma*DAi)/dKsi

def Sdcross(P_nksi : float,P_ksieta : float,Gamma : float,DAi : float, Eg : Element, Ed : Element,eta : np.ndarray):
    
    Vect = (Eg.get_grad()+Ed.get_grad())/2

    return -Gamma*(P_ksieta/P_nksi)*(Vect@eta)*DAi

# %% Geométrie du maillage et donnée

Lx = 1 #2 cm
Ly = 1

mesh_params_T = {
    'mesh_type': 'TRI',
    'lc': 0.01
}

mesh_params_Q = {
    'mesh_type': 'QUAD',
    'Nx': 25,
    'Ny': 5
}

Gamma = 0.5
q = 1000

#%% Génération du maillage
mesher = MeshGenerator(verbose=True)


mesh_objQ = mesher.rectangle([0, Lx, 0, Ly], mesh_params_Q)


conecQ = MeshConnectivity(mesh_objQ)


conecQ.compute_connectivity()

CL_paramsT = {
    0:('D',100),
    1:('N', 0),
    2:('D',200),
    3:('N',0)
}

plotter = MeshPlotter()
#plotter.plot_mesh(mesh_objQ, label_points=True, label_elements=True, label_faces=True)

#%% Stockage des données
QuadList = [Element(mesh_objQ, i) for i in range(mesh_objQ.get_number_of_elements())]

for elem in QuadList:
    #Initialisation du champ et gradient a 0 dans ts les élements pour la première itération
    elem.set_value(0)
    elem.set_grad(0,0)


Aq = np.zeros((len(QuadList),len(QuadList)))


Bq = np.zeros(len(QuadList))

GradCalculatorQ = MeanSquare(mesh_objQ,QuadList,CL_paramsT)

#%% Calcul Triangle



# On fait 4 itérations de calcul
for i in range(4):

    #Initialisation des matrices
    Aq = np.zeros((len(QuadList),len(QuadList)))
    Bq = np.zeros(len(QuadList))

    #Terme source
    for i in range(len(QuadList)):
        Bq[i] += q*QuadList[i].get_Area()

    NbBoundaryFaces = mesh_objQ.get_number_of_boundary_faces()
    NbFaces = mesh_objQ.get_number_of_faces()

    print("\n\n Iteration : {}\n\n".format((i+1)))

    for i in range(NbBoundaryFaces):
        tag = mesh_objQ.get_boundary_face_to_tag(i)
        elems = mesh_objQ.get_face_to_elements(i)
        #Que le triangle gauche, le droit n'existe pas
        Eg = QuadList[elems[0]]

        FaceNodes = mesh_objQ.get_face_to_nodes(i)

        Xa = mesh_objQ.node_to_xcoord[FaceNodes[0]]
        Ya = mesh_objQ.node_to_ycoord[FaceNodes[0]]
        
        Xb = mesh_objQ.node_to_xcoord[FaceNodes[1]]
        Yb = mesh_objQ.node_to_ycoord[FaceNodes[1]]

        DAi = np.sqrt((Xb-Xa)**2+(Yb-Ya)**2)

        if CL_paramsT[tag][0] == 'D':

            XP = Eg.ElementCoord[0]
            YP = Eg.ElementCoord[1]

            XA = (Xa+Xb)/2
            YA = (Ya+Yb)/2

            dKsi = np.sqrt((XA-XP)**2+(YA-YP)**2)

            data = dataFace(Xa,Xb,Ya,Yb,XA,XP,YA,YP)

            P_nksi = Pnksi(data,DAi,dKsi)
            P_ksieta = Pksieta(data,dKsi,DAi)

            eta = np.array([(data.Xb-data.Xa)/DAi,(data.Yb-data.Ya)/DAi])

            di = (Gamma/P_nksi)*(DAi/dKsi)

            #print("D, di = {:.3f}, dKsi = {:.3f}, DAi = {:.3f},Pnksi = {:.3f}, P_ksieta = {:.3f}".format(di,dKsi,DAi,P_nksi,P_ksieta))
        
        ##################  Pas sure du tout , j'approx (phi_b-phi_a)/dEta par gradPhi(A).eta
            sdcross = -Gamma*(P_ksieta/P_nksi)*(Eg.get_grad()@eta)*DAi
            
            Aq[elems[0],elems[0]] += di
            Bq[elems[0]] += sdcross + di*CL_paramsT[tag][1]

            
            

        elif CL_paramsT[tag][0] == 'N':
            #print("Face : {}N, Phi={}".format(i,CL_paramsT[tag][1]))
            Bq[elems[0]] += Gamma*CL_paramsT[tag][1]*DAi



    # Calcul hors fontières limites, uniquement face interne

    for i in range(NbBoundaryFaces,NbFaces):
        elems = mesh_objQ.get_face_to_elements(i)
        #Elems[0] = Triangle gauche
        #Elems[1] = Triangle droit

        Eg = QuadList[elems[0]]
        Ed = QuadList[elems[1]]

        data = FaceCoordinatesCalcultion(mesh_objQ,i)

        DAi = np.sqrt((data.Xa-data.Xb)**2+(data.Ya-data.Yb)**2)
        dKsi = np.sqrt((data.XA-data.XP)**2+(data.YA-data.YP)**2)

        P_nksi = Pnksi(data,DAi,dKsi)
        P_ksieta = Pksieta(data,dKsi,DAi)

        eta = np.array([data.Xb-data.Xa,data.Yb-data.Ya])

        di = Di(P_nksi,Gamma,DAi,dKsi)
        sdcross = Sdcross(P_nksi,P_ksieta,Gamma,DAi,Eg,Ed,eta)

        Aq[elems[0],elems[0]] += di
        Aq[elems[1],elems[1]] += di

        Aq[elems[0],elems[1]] -= di
        Aq[elems[1],elems[0]] -= di

        Bq[elems[0]] += sdcross
        Bq[elems[1]] -= sdcross

    Phi = np.linalg.solve(Aq,Bq)

    for elem_i in range(len(QuadList)):
        QuadList[elem_i].set_value(Phi[elem_i])


    #Avec les valeur du champs mise à jour, on calcule un nouveau gradient
    GradCalculatorQ.updateElements(QuadList)
    GradCalculatorQ.calculMeanSquare()


#%%

# for elem in QuadList:
#     print("Quad {} : V:{} | Grad: {}".format(elem.index,elem.get_value(),np.round(elem.get_grad()),3))

# #%%

# def champ(x,y):
#     return 100*x + 100 + 0*y

# for elem in QuadList:
#     elem.set_value(champ(elem.ElementCoord[0],elem.ElementCoord[1]))

# #%%
# GradCalculatorQ.calculMeanSquare()



# %% Color plotting

#Retreving the data for plotting
Values = np.zeros(len(QuadList))
for i in range(len(QuadList)):
    Values[i] = QuadList[i].get_value()

plotter = MeshPlotter()
nodes, elements = plotter.prepare_data_for_pyvista(mesh_objQ)
pv_mesh = pv.PolyData(nodes, elements)
pv_mesh['Champ T'] = Values

pl = pvQt.BackgroundPlotter()  # Allow the execution to continue
pl.add_mesh(pv_mesh, scalars='Champ T', show_edges=True, cmap='hot')

pl.camera_position = 'xy'
pl.show_grid()
pl.show()

#%% Solution analytique
def SolAnalytique(x,y):
    return (100/Lx + (q/(2*Gamma))*(Lx-x))*x + 100 + 0*y
# %%
def errorQuadratique():
    error = 0
    for elem in QuadList:
        Coord = elem.get_Coord()
        error += elem.get_Area()*(SolAnalytique(Coord[0],Coord[1]) - elem.get_value())**2

    return np.sqrt(error/(Lx*Ly))

# %%
