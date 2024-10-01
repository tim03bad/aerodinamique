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

def e_ksi(self):
        
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

    delta_ksi = np.sqrt((self.XA-self.XP)**2 + (self.YA-self.YP)**2)
    e_ksi[0] = (self.XA-self.XP)/delta_ksi
    e_ksi[1] = (self.YA-self.YP)/delta_ksi
        
    return e_ksi
    
def e_eta(self):
        
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
        
    delta_eta = np.sqrt((self.Xa-self.Xb)**2 + (self.Ya-self.Yb)**2)
    e_eta[0] = (self.Xb-self.Xa)/delta_eta
    e_eta[1] = (self.Yb-self.Ya)/delta_eta
        
    return e_eta
    
def normal(self):
        
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
        
    delta_A = np.sqrt((self.Xa-self.Xb)**2 + (self.Ya-self.Yb)**2)
    normal[0] = (self.Yb-self.Ya)/delta_A
    normal[1] = -(self.Xb-self.Xa)/delta_A
        
    return normal


def Pnksi(self):

    e_k = e_ksi(self)
    n = normal(self)
    
    return np.dot(e_k,n)

def Pksieta(self):

    e_k = e_ksi(self)
    e_e = e_eta(self)

    return np.dot(e_k,e_e)

def Di(P_nksi : float,Gamma : float,DAi : float,dKsi : float):

    return (1/P_nksi)*(Gamma*DAi)/dKsi

def Sdcross(P_nksi : float,P_ksieta : float,Gamma : float,DAi : float, Eg : Element, Ed : Element,eta : np.ndarray):
    
    Vect = (Eg.get_grad()+Ed.get_grad())/2

    return -Gamma*(P_ksieta/P_nksi)*(Vect@eta)*DAi

