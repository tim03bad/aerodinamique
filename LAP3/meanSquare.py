import numpy as np


from mesh import Mesh
import mesh
from meshGenerator import MeshGenerator
from meshConnectivity import MeshConnectivity
from meshPlotter import MeshPlotter


from element import Element
from champ import Champ
from CL import CL

class MeanSquare:
    def __init__(self):
        
        """
        Constructeur de l'objet MeanSquare

        Parameters
        ----------
        None

        Attributes
        ----------
        mesh_obj : Mesh
            Objet contenant les informations du maillage
        elements : list
            Liste des objets Element
        champ : Champ
            Objet contenant le champ
        CL : CL
            Objet contenant les conditions aux limites

        Returns
        -------
        None
        """
        mesher = MeshGenerator()
        mesh_parameters = {'mesh_type': 'TRI','lc': 0.5}

        self.mesh_obj = mesher.rectangle([0.0, 1.0, 0.0, 1.0], mesh_parameters)

        conec = MeshConnectivity(self.mesh_obj)
        conec.compute_connectivity()




    def createElements(self):
        
        #Construction des elements
        """
        Construction des elements du maillage et initialisation du champ

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.elements = [Element(self.mesh_obj, i) for i in range(self.mesh_obj.get_number_of_elements())] 

        #Initialisation du champ dans les elements
        self.champ.set_ElementsValue(self.elements)

    def createCL(self,parameters : dict[int,(str,float)]):
        self.CL = CL(parameters)

    def constructionATA(self):

        """
        Construction de la matrice ATA

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        
        for i in range(self.mesh_obj.get_number_of_faces()):

            elem = self.mesh_obj.get_face_to_elements(i)

            if elem[1]!= -1: #face interne
                #On récupère un accès aux éléments concernés
                Eg = self.getElement(elem[0])
                Ed = self.getElement(elem[1])

                #Calcul des Delta centre à centre
                DX = Ed.get_Coord()[0] - Eg.get_Coord()[0]
                DY = Ed.get_Coord()[1] - Eg.get_Coord()[1]

                Ald = np.array((2,2))
                Ald[0,0] = DX**2
                Ald[1,0] = DX*DY
                Ald[0,1] = DY*DX
                Ald[1,1] = DY**2

                Eg.ATA_add(Ald)
                Ed.ATA_add(Ald)

            else: #face externe (prise en compte condition au limite)
                #La classe CL, gère automatiquement les conditions au bord
                Eg.ATA_add(self.CL.calculCL_ATA(self.mesh_obj.get_face_tag(i),i,Eg))


    def constructionB(self):

        """
        Construction du second membre B

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        
        for i in range(self.mesh_obj.get_number_of_faces()):

            elem = self.mesh_obj.get_face_to_elements(i)

            if elem[1]!= -1: #face interne
                Eg = self.getElement(elem[0])
                Ed = self.getElement(elem[1])

                DXE = Ed.get_Coord()[0] - Eg.get_Coord()[0]
                DYE = Ed.get_Coord()[1] - Eg.get_Coord()[1]

                Bld = np.array(2)
                Bld[0,0] = DXE*(Ed.get_value() - Eg.get_value())
                Bld[1,0] = DYE*(Ed.get_value() - Eg.get_value())

                Eg.B_add(Bld)
                Ed.B_add(Bld)
            
            else: #face externe (prise en compte condition au limite)
                Eg.B_add(self.CL.calculCL_B(self.mesh_obj.get_face_tag(i),i,Eg))
    
    def calculMeanSquare(self):
        """
        Calcul du gradient de chaque élément par résolution du système
        linéaire ATA X = B

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for E in self.elements:
            E.calculGRAD()


    def getElement(self,index : int):
        return self.elements[index]

    def setChamp(self, champ : function, grad : function):
        self.champ = Champ(champ, grad)
    