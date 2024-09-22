import numpy as np

from mesh import Mesh
from meshGenerator import MeshGenerator
from meshConnectivity import MeshConnectivity
from meshPlotter import MeshPlotter


from element import Element
from champ import Champ
from CL import CL

class MeanSquare:
    def __init__(self,mesh_parameters):
        
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
        mesher = MeshGenerator(verbose=False)


        self.mesh_obj = mesher.rectangle([0.0, 1.0, 0.0, 1.0], mesh_parameters)

        conec = MeshConnectivity(self.mesh_obj,verbose=False)
        conec.compute_connectivity()



    def plot(self):
        plotter = MeshPlotter()
        plotter.plot_mesh(self.mesh_obj, label_points=True, label_elements=True, label_faces=True)

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

    def createCL(self,parameters : dict[int,(str,any)]):
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
            
            Eg = self.getElement(elem[0]) #Element gauche (existe toujours)

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

                #Eg.storeA(Ald,(Eg.index,Ed.index))
                #Ed.storeA(Ald,(Ed.index,Eg.index))

            else: #face externe (prise en compte condition au limite)
                #La classe CL, gère automatiquement les conditions au bord
                Eg.ATA_add(self.CL.calculCL_ATA(self.mesh_obj.get_boundary_face_to_tag(i),i,Eg))

                #Eg.storeA(self.CL.calculCL_ATA(self.mesh_obj.get_boundary_face_to_tag(i),i,Eg),(Eg.index,-1))


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


                #Eg.storeB(Bld)
                #Ed.storeB(Bld)
            
            else: #face externe (prise en compte condition au limite)
                Eg.B_add(self.CL.calculCL_B(self.mesh_obj.get_boundary_face_to_tag(i),i,Eg))

                #Eg.storeB(self.CL.calculCL_B(self.mesh_obj.get_boundary_face_to_tag(i),i,Eg))
    
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


    def calculTailleMoyenne(self):
        listOfArea = np.array([E.get_Area() for E in self.elements])

        return np.sqrt(np.sum(listOfArea**2)/len(listOfArea))


    def getElement(self,index : int):
        return self.elements[index]

    def setChamp(self, champ, grad):
        try:
            self.champ = Champ(champ, grad)

        except TypeError:
            print("Vous devez entrer des fonctions python")


    def debug(self):
        gradAnal = []
        gradNum = []

        for E in self.elements:
            gradNum.append(E.get_grad())
            Coordinate = E.get_Coord()
            print("Coord E{} : {} | Value : {}".format(E.index,Coordinate,E.get_value()))
            gradAnal.append(self.champ.grad(Coordinate[0], Coordinate[1]))

        print("Ei : Analytique | Numerique")
        for i in range(len(gradAnal)):
            print("E{} : {} | {}".format(i,gradAnal[i],gradNum[i]))

    def error(self):
        
        NormsN = np.array([E.getGradNorm() for E in self.elements])
        NormsA = np.array([np.linalg.norm(self.champ.grad(E.get_Coord()[0], E.get_Coord()[1])) for E in self.elements])

        #print(NormsA)
        #print(NormsN)



        return np.sqrt(np.sum((NormsA-NormsN)**2)/len(self.elements))

        
    