import numpy as np

from mesh import Mesh
from dataFace import dataFace
from element import Element

class Utilities:

    crossDiff = True

    def __init__(self,crossDiff = True):
        self.crossDiff = crossDiff

    @staticmethod
    def FaceCoordinatesCalcultion(mesh_obj: Mesh,face : int,PolyList : list[Element]):
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
        dXi = data.Xb - data.Xa
        dYi = data.Yb - data.Ya

        a = dYi*(data.XA-data.XP)/(DAi*dKsi)
        b = dXi*(data.YA-data.YP)/(DAi*dKsi)

        return a-b
    
    @staticmethod
    def Pksieta(data : dataFace,dKsi,DAi):

        a = (data.XA-data.XP)*(data.Xa-data.Xb)/(dKsi*DAi)
        b = (data.YA-data.YP)*(data.Ya-data.Yb)/(dKsi*DAi)

        return a+b 
    
    @staticmethod
    def Di(P_nksi : float,Gamma : float,DAi : float,dKsi : float):

        return (1/P_nksi)*(Gamma*DAi)/dKsi
    
    @staticmethod
    def Sdcross(P_nksi : float,P_ksieta : float,Gamma : float,DAi : float, Eg : Element, Ed : Element,eta : np.ndarray):
    
        Vect = (Eg.get_grad()+Ed.get_grad())/2

        if Utilities.crossDiff:
            return -Gamma*(P_ksieta/P_nksi)*(Vect@eta)*DAi
        else:
            return 0
    
    @staticmethod
    def pause():
        input("Appuyer sur une touche pour continuer...")