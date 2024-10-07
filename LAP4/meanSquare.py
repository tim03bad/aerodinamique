import numpy as np
from regex import E

from mesh import Mesh
from meshGenerator import MeshGenerator
from meshConnectivity import MeshConnectivity
from meshPlotter import MeshPlotter


from element import Element
from CL import CL

class MeanSquare:
    def __init__(self,mesh_obj : Mesh,ElementsList: list[Element],CL_parameters : dict[int,(str,any)]):

        """
        Constructeur de l'objet MeanSquare

        Parameters
        ----------
        mesh_obj : Mesh
            Objet contenant les informations du maillage
        CL_parameters : dict[int,(str,any)]
            Dictionnaire qui contient les param tres de la condition au bord pour chaque face.
            Les cl s sont les tags de chaque bord et les valeurs sont des tuples (str, float) contenant
            le type de condition ('D' pour Dirichlet, 'N' pour Neumann, 'L' pour Libre) et la valeur
            de la condition (par exemple, la valeur du champ pour une condition de Dirichlet).
            Ex : param[1]=('D',2)
        champ : function
            Fonction qui prend en argument deux valeurs (x,y) et qui renvoie la valeur du champ au point (x,y)
        grad : function
            Fonction qui prend en argument deux valeurs (x,y) et qui renvoie le gradient du champ au point (x,y)

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
        self.mesh_obj = mesh_obj

        self.CL = CL(CL_parameters)

        self.elements = ElementsList



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
            
            Eg = self.getElement(elem[0]) #Element gauche (existe toujours)
           
            #print(elem)


            if elem[1]!= -1: #face interne => élements droit et gauche
        
                
                Ed = self.getElement(elem[1]) #Element droit si existe

                #Calcul des Delta centre à centre
                DX = Ed.get_Coord()[0] - Eg.get_Coord()[0]
                DY = Ed.get_Coord()[1] - Eg.get_Coord()[1]

                Ald = np.zeros((2,2))
                Ald[0,0] = DX**2
                Ald[1,0] = DX*DY
                Ald[0,1] = DY*DX
                Ald[1,1] = DY**2

                Eg.ATA_add(Ald)
                Ed.ATA_add(Ald)

                Eg.storeA(Ald,(Eg.index,Ed.index))
                Ed.storeA(Ald,(Ed.index,Eg.index))

            else: #face externe (prise en compte condition au limite)
                #La classe CL, gère automatiquement les conditions au bord
                Ald = self.CL.calculCL_ATA(self.mesh_obj.get_boundary_face_to_tag(i),i,Eg) 

                Eg.ATA_add(Ald)

                #Debugage, logging des matrices ajouté dans la matrice ATA
                Eg.storeA(Ald,(Eg.index,-1))



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

            Eg = self.getElement(elem[0]) #Element gauche (existe toujours)

            if elem[1]!= -1: #face interne

                Ed = self.getElement(elem[1])

                DXE = Ed.get_Coord()[0] - Eg.get_Coord()[0]
                DYE = Ed.get_Coord()[1] - Eg.get_Coord()[1]

                Bld = np.zeros(2)
                
                Bld[0] = DXE*(Ed.get_value() - Eg.get_value())
                Bld[1] = DYE*(Ed.get_value() - Eg.get_value())

                Eg.B_add(Bld)
                Ed.B_add(Bld)


                Eg.storeB(Bld)
                Ed.storeB(Bld)
            
            else: #face externe (prise en compte condition au limite)
                Bld = self.CL.calculCL_B(self.mesh_obj.get_boundary_face_to_tag(i),i,Eg)

                Eg.B_add(Bld)

                #Debugage, logging des matrices ajouté dans le second membre
                Eg.storeB(Bld)
    
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
            E.resetMatrix()
        
        self.constructionATA()
        self.constructionB()
        for E in self.elements:
            E.calculGRAD()


    def calculTailleMoyenne(self):
        
        """
        Calcul de la taille moyenne des éléments.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        float
            Taille moyenne des éléments.
        """
        listOfArea = np.array([E.get_Area() for E in self.elements])

        return np.sqrt(np.sum(listOfArea**2)/len(listOfArea))



###############################
    def getElement(self,index):
        """
        Retourne l'élément d'index donné.
        
        Parameters
        ----------
        index : int
            Index de l'élément.
        
        Returns
        -------
        Element
            L'élément d'index donné.
        
        Notes
        -----
        None
        """
        return self.elements[index]
