#%% Import


from math import gamma
import numpy as np
from mesh import Mesh
from champ import Champ
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

def e_ksi(data : dataFace):
        
    """
    Construction du vecteur e_ksi

    Parameters
    ----------
    data : dataFace
        Coordonnées de la face

    Returns
    -------
    ndarray
        Composantes du vecteur e_ksi
    """
        
    e_ksi = np.zeros(2)

    delta_ksi = np.sqrt((data.XA-data.XP)**2 + (data.YA-data.YP)**2)
    e_ksi[0] = (data.XA-data.XP)/delta_ksi
    e_ksi[1] = (data.YA-data.YP)/delta_ksi
        
    return e_ksi
    
def e_eta(data : dataFace):
        
    """
    Construction du vecteur e_eta

    Parameters
    ----------
    data : dataFace
        Coordonnées de la face

    Returns
    -------
    ndarray
        Composantes du vecteur e_eta
    """
        
    e_eta = np.zeros(2)
        
    delta_eta = np.sqrt((data.Xa-data.Xb)**2 + (data.Ya-data.Yb)**2)
    e_eta[0] = (data.Xb-data.Xa)/delta_eta
    e_eta[1] = (data.Yb-data.Ya)/delta_eta
        
    return e_eta
    
def normal(data : dataFace):
        
    """
    Construction du vecteur normalisé

    Parameters
    ----------
    data : dataFace
        Coordonnées de la face

    Returns
    -------
    ndarray
    Composantes du vecteur normalisé
    """
        
    normal = np.zeros(2)
        
    delta_A = np.sqrt((data.Xa-data.Xb)**2 + (data.Ya-data.Yb)**2)
    normal[0] = (data.Yb-data.Ya)/delta_A
    normal[1] = -(data.Xb-data.Xa)/delta_A
        
    return normal


def Pnksi(data : dataFace):

    e_k = e_ksi(data)
    n = normal(data)
    
    return np.dot(e_k,n)

def Pksieta(data : dataFace):

    e_k = e_ksi(data)
    e_e = e_eta(data)

    return np.dot(e_k,e_e)

def Di(P_nksi : float,Gamma : float,DAi : float,dKsi : float):

    return (1/P_nksi)*(Gamma*DAi)/dKsi

def Sdcross(P_nksi : float,P_ksieta : float,Gamma : float,DAi : float, Eg : Element, Ed : Element,eta : np.ndarray):
    
    Vect = (Eg.get_grad()+Ed.get_grad())/2

    return -Gamma*(P_ksieta/P_nksi)*(Vect@eta)*DAi

# %% Geométrie du maillage et donnée
'''Maillage grossier'''
Lx = 0.02
Ly = 0.02

mesh_params_T_grossier = {
    'mesh_type': 'TRI',
    'lc': 0.008
}

mesh_params_Q_grossier = {
    'mesh_type': 'QUAD',
    'Nx': 5,
    'Ny': 5
}

'''Maillage fin'''
Lx = 0.02
Ly = 0.02

mesh_params_T_fin = {
    'mesh_type': 'TRI',
    'lc': 0.0008
}

mesh_params_Q_fin = {
    'mesh_type': 'QUAD',
    'Nx': 20,
    'Ny': 20
}

'''constantes'''
Gamma = 0.5
q = 1000e3
Ta = 100
Tb = 200

#%% Génération du maillage
mesher = MeshGenerator(verbose=True)

'''maillage grossier'''

mesh_objT_grossier = mesher.rectangle([0, Lx, 0, Ly], mesh_params_T_grossier)
mesh_objQ_grossier = mesher.rectangle([0, Lx, 0, Ly], mesh_params_Q_grossier)

conecT_grossier = MeshConnectivity(mesh_objT_grossier)
conecQ_grossier = MeshConnectivity(mesh_objQ_grossier)

conecT_grossier.compute_connectivity()
conecQ_grossier.compute_connectivity()

'''maillage fin'''

mesh_objT_fin = mesher.rectangle([0, Lx, 0, Ly], mesh_params_T_fin)
mesh_objQ_fin = mesher.rectangle([0, Lx, 0, Ly], mesh_params_Q_fin)

conecT_fin = MeshConnectivity(mesh_objT_fin)
conecQ_fin = MeshConnectivity(mesh_objQ_fin)

conecT_fin.compute_connectivity()
conecQ_fin.compute_connectivity()

CL_paramsT = {
    0:('D',Ta),
    1:('N', 0),
    2:('D',Tb),
    3:('N',0)
}

'''plotter = MeshPlotter()
plotter.plot_mesh(mesh_objT_grossier, label_points=True, label_elements=True, label_faces=True)
plotter.plot_mesh(mesh_objQ_grossier, label_points=True, label_elements=True, label_faces=True)
plotter.plot_mesh(mesh_objT_fin, label_points=True, label_elements=True, label_faces=True)
plotter.plot_mesh(mesh_objQ_fin, label_points=True, label_elements=True, label_faces=True)'''

#%% Stockage des données
def fct(x : float, y : float):
    return ((Tb-Ta)/Lx + (Lx-x)*q/(2*Gamma))*x + Ta

def grad(x : float, y : float):
    Grad = np.zeros(2)
    Grad[0] = (Tb-Ta)/Lx + (Lx-2*x)*q/(2*Gamma)
    return Grad

'''liste grossière'''

TriangleList_grossier = [Element(mesh_objT_grossier, i) for i in range(mesh_objT_grossier.get_number_of_elements())]
QuadList_grossier = [Element(mesh_objQ_grossier, i) for i in range(mesh_objQ_grossier.get_number_of_elements())]

for elem in TriangleList_grossier:
    #Initialisation du champ et gradient a 0 dans ts les élements pour la première itération
    elem.set_value(0)
    elem.set_grad(0,0)


for elem in QuadList_grossier:
    #Initialisation du champ et gradient a 0 dans ts les élements pour la première itération
    elem.set_value(0)
    elem.set_grad(0,0)
    
GradCalculatorQ_grossier = MeanSquare(mesh_objQ_grossier ,QuadList_grossier ,CL_paramsT)
GradCalculatorT_grossier = MeanSquare(mesh_objT_grossier ,TriangleList_grossier ,CL_paramsT)
    
'''liste fine'''

TriangleList_fin = [Element(mesh_objT_fin, i) for i in range(mesh_objT_fin.get_number_of_elements())]
QuadList_fin = [Element(mesh_objQ_fin, i) for i in range(mesh_objQ_fin.get_number_of_elements())]

for elem in TriangleList_fin:
    #Initialisation du champ et gradient a 0 dans ts les élements pour la première itération
    elem.set_value(0)
    elem.set_grad(0,0)


for elem in QuadList_fin:
    #Initialisation du champ et gradient a 0 dans ts les élements pour la première itération
    elem.set_value(0)
    elem.set_grad(0,0)
    
GradCalculatorQ_fin = MeanSquare(mesh_objQ_fin ,QuadList_fin ,CL_paramsT)
GradCalculatorT_fin = MeanSquare(mesh_objT_fin ,TriangleList_fin ,CL_paramsT)


#%% Calcul Carré

# On fait 4 itérations de calcul
def Phi(mesh_objQ : Mesh, CL_paramsT : dict[int,(str,any)], QuadList : list, GradCalculatorQ : MeanSquare):
    
    Aq = np.zeros((len(QuadList),len(QuadList)))
    Bq = np.zeros(len(QuadList))
    
    for i in range(4):
        NbBoundaryFaces = mesh_objQ.get_number_of_boundary_faces()
        NbFaces = mesh_objQ.get_number_of_faces()
    
        print("\n\n Iteration : {}\n\n".format(i))
    
        for i in range(NbBoundaryFaces):
            tag = mesh_objQ.get_boundary_face_to_tag(i)
            elems = mesh_objQ.get_face_to_elements(i)
            #Que le triangle gauche, le droit n'existe pas
            Eg = QuadList[elems[0]]
    
            FaceNodes = mesh_objQ.get_face_to_nodes(i)
    
            Xa = mesh_objQ.node_to_xcoord[FaceNodes[0]]
            Xb = mesh_objQ.node_to_xcoord[FaceNodes[1]]
    
            Ya = mesh_objQ.node_to_ycoord[FaceNodes[0]]
            Yb = mesh_objQ.node_to_ycoord[FaceNodes[1]]
    
            DAi = np.sqrt((Xb-Xa)**2+(Yb-Ya)**2)
    
            if CL_paramsT[tag][0] == 'D':
    
                XA = Eg.ElementCoord[0]
                YA = Eg.ElementCoord[1]
    
                XP = (Xa+Xb)/2
                YP = (Ya+Yb)/2
    
                dKsi = np.sqrt((XA-XP)**2+(YA-YP)**2)
    
                data = dataFace(Xa,Xb,Ya,Yb,XA,XP,YA,YP)
    
                P_nksi = Pnksi(data)
                P_ksieta = Pksieta(data)
    
                eta = e_eta(data)
    
                di = (Gamma/P_nksi)*(DAi/dKsi)
            
            ##################  Pas sure du tout , j'approx phi_b-phi_a par gradPhi(A).eta
                sdcross = -Gamma*(P_ksieta/P_nksi)*(Eg.get_grad()@eta)*DAi
                
                Aq[elems[0],elems[0]] += di
                Bq[elems[0]] += sdcross + di*CL_paramsT[tag][1]
                
    
            elif CL_paramsT[tag][0] == 'N':
                Bq[elems[0]] += Gamma*CL_paramsT[tag][1]*DAi
    
    
    
        # Calcul hors fontières limites, uniquement face interne
    
        for i in range(NbBoundaryFaces,NbFaces):
            elems = mesh_objQ.get_face_to_elements(i)
            #Elems[0] = Triangle gauche
            #Elems[1] = Triangle droit
    
            Eg = QuadList[elems[0]]
            Ed = QuadList[elems[1]]
    
            data = FaceCoordinatesCalcultion(mesh_objQ,Eg,Ed,i)
    
            DAi = np.sqrt((data.Xa-data.Xb)**2+(data.Ya-data.Yb)**2)
            dKsi = np.sqrt((data.XA-data.XP)**2+(data.YA-data.YP)**2)
    
            P_nksi = Pnksi(data)
            P_ksieta = Pksieta(data)
    
            eta = e_eta(data)
    
            di = Di(P_nksi,Gamma,DAi,dKsi)
            sdcross = Sdcross(P_nksi,P_ksieta,Gamma,DAi,Eg,Ed,eta)
    
            Aq[elems[0],elems[0]] += di
            Aq[elems[1],elems[1]] += di
    
            Aq[elems[0],elems[1]] -= di
            Aq[elems[1],elems[0]] -= di
    
            Bq[elems[0]] += sdcross
            Bq[elems[1]] -= sdcross
        
        for i in range(len(QuadList)):
            elem = QuadList[i]
            S = elem.get_Area()
            Bq[i] -= q*S
    
        Phi = np.linalg.solve(Aq,Bq)
    
        for elem_i in range(len(QuadList)):
            QuadList[elem_i].set_value(Phi[elem_i])
    
        for elem in QuadList:
            print("Valeur du carré {} : {}".format(elem.index,elem.get_value()))
    
        #Avec les valeur du champs mise à jour, on calcule un nouveau gradient
        GradCalculatorQ.updateElements(QuadList)
        GradCalculatorQ.calculMeanSquare()
    return Phi

#%%
'''
for elem in List:
    print("Triangle {} : {}".format(elem.index,elem.get_grad()))
'''
    
#%%Solution analytique

def Phi_ana(List : list) :
    Phi_ana = np.zeros(len(List))
    for i in range(len(List)):
        elem = List[i]
        Phi_ana[i] = fct(elem.ElementCoord[0],elem.ElementCoord[1])
    return Phi_ana

#%%Erreur quadratique

def erreur_quadratique(Phi, Phi_ana):
    erreur = 0
    for i in range(len(Phi)):
        erreur += (Phi[i] - Phi_ana[i])**2
    return np.sqrt(erreur/len(Phi))

#%%Calcul de la convergence

Phi_grossier = Phi(mesh_objQ_grossier, CL_paramsT, QuadList_grossier, GradCalculatorQ_grossier)
Phi_ana_grossier = Phi_ana(QuadList_grossier)

Phi_fin = Phi(mesh_objQ_fin, CL_paramsT, QuadList_fin, GradCalculatorQ_fin)
Phi_ana_fin = Phi_ana(QuadList_fin)

h_fin = GradCalculatorQ_fin.calculTailleMoyenne()
h_grossier =  GradCalculatorQ_grossier.calculTailleMoyenne()

E_fine = erreur_quadratique(Phi_fin, Phi_ana_fin)
E_grossière = erreur_quadratique(Phi_grossier, Phi_ana_grossier)

print("Ordre de convergence : ", np.log(E_grossière/E_fine)/np.log(h_grossier/h_fin))

