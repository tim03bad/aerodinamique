from re import U
from xmlrpc.client import boolean
from networkx import selfloop_edges
import numpy as np

from mesh import Mesh
from dataFace import dataFace
from element import Element

class Utilities:


    @staticmethod
    def FaceCoordinatesCalcultion(mesh_obj: Mesh,face : int,PolyList : list[Element]):

        """
        Calcul des coordonnes des sommets de la face et les centres des élements

        Parameters
        ----------
        mesh_obj : Mesh
            Objet contenant les informations du maillage
        face : int
            Numéro de la face
        PolyList : list[Element]
            Liste des  l ments du maillage

        Returns
        -------
        dataFace
            Objet contenant les coordonnées des sommets de la face et les centres des élements
        """

        FaceNodes = mesh_obj.get_face_to_nodes(face)
        Elems = mesh_obj.get_face_to_elements(face)
        Eg = PolyList[Elems[0]]
        Ed = PolyList[Elems[1]]
        
        Xa = mesh_obj.node_to_xcoord[FaceNodes[0]]
        Xb = mesh_obj.node_to_xcoord[FaceNodes[1]]

        Ya = mesh_obj.node_to_ycoord[FaceNodes[0]]
        Yb = mesh_obj.node_to_ycoord[FaceNodes[1]]

        XP = Eg.ElementCoord[0]
        XA = Ed.ElementCoord[0]

        YP = Eg.ElementCoord[1]
        YA = Ed.ElementCoord[1]

        return dataFace(Xa,Xb,Ya,Yb,XA,XP,YA,YP)
    
    @staticmethod
    def Pnksi(data : dataFace,DAi :float,dKsi : float):
        """
        Calcul de la quantité P_nksi

        Parameters
        ----------
        data : dataFace
            Objet contenant les coordonnées des sommets de la face et les centres des élements
        DAi : float
            Longueur de la face
        dKsi : float
            Longueur entre le centre de l'élément gauche et le centre de l'élément droit

        Returns
        -------
        float
            P_nksi
        """
        dXi = data.Xb - data.Xa
        dYi = data.Yb - data.Ya

        a = dYi*(data.XA-data.XP)/(DAi*dKsi)
        b = dXi*(data.YA-data.YP)/(DAi*dKsi)

        return a-b
    
    @staticmethod
    def Pksieta(data : dataFace,dKsi,DAi):

        """
        Calcul de la quantité P_ksieta

        Parameters
        ----------
        data : dataFace
            Objet contenant les coordonnées des sommets de la face et les centres des élements
        dKsi : float
            Longueur entre le centre de l'élément gauche et le centre de l'élément droit
        DAi : float
            Longueur de la face

        Returns
        -------
        float
            P_ksieta
        """
        a = (data.XA-data.XP)*(data.Xa-data.Xb)/(dKsi*DAi)
        b = (data.YA-data.YP)*(data.Ya-data.Yb)/(dKsi*DAi)

        return a+b 
    
    @staticmethod
    def Di(P_nksi : float,Gamma : float,DAi : float,dKsi : float):

        """
        Calcul de la diffusion normale

        Parameters
        ----------
        P_nksi : float
            Quantité P_nksi
        Gamma : float
            Coefficient de diffusion
        DAi : float
            Longueur de la face
        dKsi : float
            Longueur entre le centre de l'élément gauche et le centre de l'élément droit

        Returns
        -------
        float
            Quantité Di
        """
        return (1/P_nksi)*(Gamma*DAi)/dKsi
    
    @staticmethod
    def Sdcross(P_nksi : float,P_ksieta : float,Gamma : float,DAi : float, Eg : Element, Ed : Element,eta : np.ndarray, cD=True ):
    
        """
        Calcul du terme de diffusion croisée

        Parameters
        ----------
        P_nksi : float
            Quantité P_nksi
        P_ksieta : float
            Quantité P_ksieta
        Gamma : float
            Coefficient de diffusion
        DAi : float
            Longueur de la face
        Eg : Element
            El ment gauche
        Ed : Element
            El ment droit
        eta : ndarray
            Vecteur unitaire normal la face
        cD : boolean, optionnel
            Prise en compte ou non du coefficient de diffusion croisée

        Returns
        -------
        float
            Terme de diffusion crois e
        """
        Vect = (Eg.get_grad()+Ed.get_grad())/2

        if cD == True:
            return -Gamma*(P_ksieta/P_nksi)*(Vect@eta)*DAi
        else:
            return 0
    
    @staticmethod
    def pause():
        input("Appuyer sur une touche pour continuer...")