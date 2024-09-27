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
        
    def e_ksi(self, i_face):
        
        """
        Construction du vecteur e_ksi

        Parameters
        ----------
        i_face : int
            Identifiant de la face

        Returns
        -------
        ndarray
            Composantes du vecteur e_ksi
        """
        
        e_ksi = np.zeros(2)
        
        elem = self.mesh_obj.get_face_to_elements(i_face)
        
        XA, YA = elem[0].get_Coord()[0], elem[0].get_Coord()[1]
        XP, YP = elem[1].get_Coord()[0], elem[1].get_Coord()[1]
        
        delta_ksi = np.sqrt((XA-XP)**2 + (YA-YP)**2)
        e_ksi[0] = (XA-XP)/delta_ksi
        e_ksi[1] = (YA-YP)/delta_ksi
        
        return e_ksi
    
    def e_eta(self, i_face):
        
        """
        Construction du vecteur e_eta

        Parameters
        ----------
        i_face : int
            Identifiant de la face

        Returns
        -------
        ndarray
            Composantes du vecteur e_eta
        """
        
        e_eta = np.zeros(2)
        
        nodes = self.mesh_obj.get_faces_to_nodes(i_face)

        xa, ya = nodes[0].get_Coord()[0], nodes[0].get_Coord()[1]
        xb, yb = nodes[1].get_Coord()[0], nodes[1].get_Coord()[1]
        
        delta_eta = np.sqrt((xa-xb)**2 + (ya-yb)**2)
        e_eta[0] = (xa-xb)/delta_eta
        e_eta[1] = (ya-yb)/delta_eta
        
        return e_eta
    
    def normal(self, i_face):
        
        """
        Construction du vecteur normalisé

        Parameters
        ----------
        i_face : int
            Identifiant de la face

        Returns
        -------
        ndarray
            Composantes du vecteur normalisé
        """
        
        normal = np.zeros(2)
        
        nodes = self.mesh_obj.get_faces_to_nodes(i_face)

        xa, ya = nodes[0].get_Coord()[0], nodes[0].get_Coord()[1]
        xb, yb = nodes[1].get_Coord()[0], nodes[1].get_Coord()[1]
        
        delta_A = np.sqrt((xa-xb)**2 + (ya-yb)**2)
        normal[0] = (ya-yb)/delta_A
        normal[1] = -(xa-xb)/delta_A
        
        return normal
    
    def constructionA(self, gamma):

        """
        Construction de la matrice A

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        
        nb_face = self.mesh_obj.get_number_of_faces()
        A = np.zeros((nb_face,nb_face))        
        
        e_ksi = np.zeros(2)
        normal = np.zeros(2)
        
        for i in range(nb_face):

            elem = self.mesh_obj.get_face_to_elements(i)
            nodes = self.mesh_obj.get_faces_to_nodes(i)
            
            Eg = self.getElement(elem[0]) #Element gauche (existe toujours)
            
            e_ksi = e_ksi(i)
            normal = normal(i)
            
            PNKSI = np.dot(normal, e_ksi)
            
            if elem[1]!= -1: #face interne => élements droit et gauche
                
                Ed = self.getElement(elem[1]) #Element droit si existe

                #Calcul des Delta centre à centre
                XA, YA = Ed.get_Coord()[0], Ed.get_Coord()[1]
                XP, YP = Eg.get_Coord()[0], Eg.get_Coord()[1]
                
                delta_ksi = np.sqrt((XA-XP)**2 + (YA-YP)**2)
                
                xa, ya = nodes[0].get_Coord()[0], nodes[0].get_Coord()[1]
                xb, yb = nodes[1].get_Coord()[0], nodes[1].get_Coord()[1]
                
                delta_A = np.sqrt((xa-xb)**2 + (ya-yb)**2)
                
                Di = gamma/PNKSI*delta_A/delta_ksi
                j = Ed.index
                k = Eg.index

                A[k,k] += Di
                A[j,j] += Di
                A[k,j] -= Di
                A[j,k] -= Di

                #Eg.storeA(Ald,(Eg.index,Ed.index))
                #Ed.storeA(Ald,(Ed.index,Eg.index))

            else: #face externe (prise en compte condition au limite)
                #La classe CL, gère automatiquement les conditions au bord
                Eg.ATA_add(self.CL.calculCL_ATA(self.mesh_obj.get_boundary_face_to_tag(i),i,Eg))

                #Eg.storeA(self.CL.calculCL_ATA(self.mesh_obj.get_boundary_face_to_tag(i),i,Eg),(Eg.index,-1))

    def constructionB(self, gamma):

        """
        Construction du second membre B

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        
        nb_face = self.mesh_obj.get_number_of_faces()
        B = np.zeros((nb_face,nb_face)) 
        
        e_ksi = np.zeros(2)
        e_eta = np.zeros(2)
        normal = np.zeros(2)
        
        for i in range(nb_face):

            elem = self.mesh_obj.get_face_to_elements(i)
            nodes = self.mesh_obj.get_faces_to_nodes(i)
            
            Eg = self.getElement(elem[0]) #Element gauche (existe toujours)
            
            e_ksi = e_ksi(i)
            e_eta = e_eta(i)
            normal = normal(i)
            
            PNKSI = np.dot(normal, e_ksi)
            PKSIETA = np.dot(e_eta, e_ksi)

            if elem[1]!= -1: #face interne
                Ed = self.getElement(elem[1])

                xa, ya = nodes[0].get_Coord()[0], nodes[0].get_Coord()[1]
                xb, yb = nodes[1].get_Coord()[0], nodes[1].get_Coord()[1]
                
                phia, phib = self.champ.grad(xa, ya), self.champ.grad(xb, yb)
                
                Sdci = -gamma*(phib-phia)*(PKSIETA/PNKSI)
                
                j = Ed.index
                k = Eg.index

                B[k] += Sdci
                B[j] -= Sdci
                #Eg.storeB(Bld)
                #Ed.storeB(Bld)
            
            else: #face externe (prise en compte condition au limite)
                Eg.B_add(self.CL.calculCL_B(self.mesh_obj.get_boundary_face_to_tag(i),i,Eg))

                #Eg.storeB(self.CL.calculCL_B(self.mesh_obj.get_boundary_face_to_tag(i),i,Eg))


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
        
        NormsN2 = np.array(np.linalg.norm(np.array([E.get_grad()-self.champ.grad(E.get_Coord()[0], E.get_Coord()[1]) for E in self.elements]))**2)
        


        return np.sqrt(np.sum(NormsN2)/len(self.elements))

        
    