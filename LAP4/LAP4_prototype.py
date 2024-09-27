#%% Import


from math import gamma
import numpy as np
from mesh import Mesh
from meshConnectivity import MeshConnectivity
from meshGenerator import MeshGenerator
from meshPlotter import MeshPlotter

from meanSquare import MeanSquare
from element import Element

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

#%% Fonctions

def FaceCoordinatesCalcultion(mesh_obj: Mesh,Eg : Element,Ed : Element ,face : int):
    FaceNodes = mesh_obj.get_face_to_nodes(face)
    Elems = mesh_obj.get_face_to_elements(face)
    
    Xa = mesh_obj.node_to_xcoord[FaceNodes[0]]
    Xb = mesh_obj.node_to_xcoord[FaceNodes[1]]

    Ya = mesh_obj.node_to_ycoord[FaceNodes[0]]
    Yb = mesh_obj.node_to_ycoord[FaceNodes[1]]

    XA = Eg.ElementCoord[0]
    XP = Ed.ElementCoord[0]

    YA = Eg.ElementCoord[1]
    YP = Ed.ElementCoord[1]

    return dataFace(Xa,Xb,Ya,Yb,XA,XP,YA,YP)


def Pnksi(data : dataFace,DAi :float,dKsi : float):

    
    dXi = data.Xb - data.Xa
    dYi = data.Yb - data.Ya

    a = dYi*(data.XA-data.XP)/(DAi*dKsi)
    b = dXi*(data.YA-data.YP)/(DAi*dKsi)

    return a-b

def Pksieta(data : dataFace,dEta,DAi):

    a = (data.XA-data.XP)*(data.Xa-data.Xb)/(dEta*DAi)
    b = (data.YA-data.YP)*(data.Ya-data.Yb)/(dEta*DAi)

    return a+b 

def Di(P_nksi : float,Gamma : float,DAi : float,dKsi : float):

    return (1/P_nksi)*(Gamma*DAi)/dKsi

def Sdcross(P_nksi : float,P_ksieta : float,Gamma : float,DAi : float, Eg : Element, Ed : Element,eta : np.ndarray):
    
    Vect = (Eg.get_grad()+Ed.get_grad())/2

    return -Gamma*(P_ksieta/P_nksi)*(Vect@eta)*DAi

# %% Geométrie du maillage et donnée

Lx = 1
Ly = 1

mesh_params_T = {
    'mesh_type': 'TRI',
    'lc': 0.4
}

mesh_params_Q = {
    'mesh_type': 'QUAD',
    'Nx': 10,
    'Ny': 10
}

Gamma = 0.5

#%% Génération du maillage
mesher = MeshGenerator(verbose=True)

mesh_objT = mesher.rectangle([0, Lx, 0, Ly], mesh_params_T)
mesh_objQ = mesher.rectangle([0, Lx, 0, Ly], mesh_params_Q)

conecT = MeshConnectivity(mesh_objT)
conecQ = MeshConnectivity(mesh_objQ)

conecT.compute_connectivity()
conecQ.compute_connectivity()

CL_paramsT = {
    0:('D',100),
    1:('N', 0),
    2:('D',200),
    3:('N',0)
}

plotter = MeshPlotter()
plotter.plot_mesh(mesh_objT, label_points=True, label_elements=True, label_faces=True)

#%% Stockage des données
TriangleList = [Element(mesh_objT, i) for i in range(mesh_objT.get_number_of_elements())]
QuadList = [Element(mesh_objQ, i) for i in range(mesh_objQ.get_number_of_elements())]

for elem in TriangleList:
    #Initialisation du champ et gradient a 0 dans ts les élements pour la première itération
    elem.set_value(0)
    elem.set_grad(0,0)

At = np.zeros((len(TriangleList),len(TriangleList)))
Aq = np.zeros((len(QuadList),len(QuadList)))

Bt = np.zeros(len(TriangleList))
Bq = np.zeros(len(QuadList))

GradCalculatorT = MeanSquare(mesh_objT,TriangleList,CL_paramsT)

#%% Calcul Triangle

# On fait 4 itérations de calcul
for i in range(1):
    NbBoundaryFaces = mesh_objT.get_number_of_boundary_faces()
    NbFaces = mesh_objT.get_number_of_faces()

    print("\n\n Iteration : {}\n\n".format(i))

    for i in range(NbBoundaryFaces):
        tag = mesh_objT.get_boundary_face_to_tag(i)
        elems = mesh_objT.get_face_to_elements(i)
        #Que le triangle gauche, le droit n'existe pas
        Eg = TriangleList[elems[0]]

        FaceNodes = mesh_objT.get_face_to_nodes(i)

        Xa = mesh_objT.node_to_xcoord[FaceNodes[0]]
        Xb = mesh_objT.node_to_xcoord[FaceNodes[1]]

        Ya = mesh_objT.node_to_ycoord[FaceNodes[0]]
        Yb = mesh_objT.node_to_ycoord[FaceNodes[1]]

        DAi = np.sqrt((Xb-Xa)**2+(Yb-Ya)**2)

        if CL_paramsT[tag][0] == 'D':

            XA = Eg.ElementCoord[0]
            YA = Eg.ElementCoord[1]

            XP = (Xa+Xb)/2
            YP = (Ya+Yb)/2

            dKsi = np.sqrt((XA-XP)**2+(YA-YP)**2)

            data = dataFace(Xa,Ya,Xb,Yb,XA,YA,XP,YP)

            P_nksi = Pnksi(data,DAi,dKsi)
            P_ksieta = Pksieta(data,DAi,dKsi)

            eta = np.array([data.Xb-data.Xa,data.Yb-data.Ya])

            di = (Gamma/P_nksi)*(DAi/dKsi)
        
        ##################  Pas sure du tout , j'approx phi_b-phi_a par gradPhi(A).eta
            sdcross = -Gamma*(P_ksieta/P_nksi)*(Eg.get_grad()@eta)*DAi
            
            At[elems[0],elems[1]] += di
            Bt[elems[0]] += sdcross + di*CL_paramsT[tag][1]
            

        elif CL_paramsT[tag][0] == 'N':
            Bt[elems[0]] += Gamma*CL_paramsT[tag][1]*DAi



    # Calcul hors fontières limites, uniquement face interne

    for i in range(NbBoundaryFaces,NbFaces):
        elems = mesh_objT.get_face_to_elements(i)
        #Elems[0] = Triangle gauche
        #Elems[1] = Triangle droit

        Eg = TriangleList[elems[0]]
        Ed = TriangleList[elems[1]]

        data = FaceCoordinatesCalcultion(mesh_objT,Eg,Ed,i)

        DAi = np.sqrt((data.Xa-data.Xb)**2+(data.Ya-data.Yb)**2)
        dKsi = np.sqrt((data.XA-data.XP)**2+(data.YA-data.YP)**2)

        P_nksi = Pnksi(data,DAi,dKsi)
        P_ksieta = Pksieta(data,dKsi,DAi)

        eta = np.array([data.Xb-data.Xa,data.Yb-data.Ya])

        di = Di(P_nksi,Gamma,DAi,dKsi)
        sdcross = Sdcross(P_nksi,P_ksieta,Gamma,DAi,Eg,Ed,eta)

        At[elems[0],elems[0]] += di
        At[elems[1],elems[1]] += di

        At[elems[0],elems[1]] -= di
        At[elems[1],elems[0]] -= di

        Bt[elems[0]] += sdcross
        Bt[elems[1]] += sdcross

    Phi = np.linalg.solve(At,Bt)

    for elem_i in range(len(TriangleList)):
        TriangleList[elem_i].set_value(Phi[elem_i])

    for elem in TriangleList:
        print("Valeur du triangle {} : {}".format(elem.index,elem.get_value()))

    #Avec les valeur du champs mise à jour, on calcule un nouveau gradient
    GradCalculatorT.updateElements(TriangleList)
    GradCalculatorT.calculMeanSquare()


#%%
for elem in TriangleList:
    print("Triangle {} : {}".format(elem.index,elem.get_grad()))




