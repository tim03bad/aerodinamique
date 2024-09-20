#Import de class
from mesh import Mesh

#Import bibliotheques
import numpy as np




class Element():

    def __init__(self, mesh_obj :Mesh, index):
        """
        Constructeur de l'objet Element

        Parameters
        ----------
        mesh_obj : Mesh
            Objet contenant les informations du maillage
        index : int
            Index de l'élément dans le maillage

        Attributes
        ----------
        index : int
            Index de l'élément dans le maillage
        value : float
            Valeur du champ au centre de l'élément
        grad : ndarray
            Gradient du champ au centre de l'élément
        ATA : ndarray
            Matrice ATA de l'élément
        B : ndarray
            Matrice B de l'élément
        nodes : list
            Liste des noeuds de l'élément
        nodesCoord : dict
            Dictionnaire Node : Coordonnées
        ElementCoord : ndarray
            Coordonnées du centre de l'élément

        """
        
        self.index = index

        #Grandeurs du champs dans l'élément
        self.value = 0
        self.grad  = np.zeros(2)

        #Matrices de l'élement
        self.ATA = np.zeros((2,2))
        self.B = np.zeros(2)

        #Calcul des coordonnées de l'élément
        self.nodes = mesh_obj.get_element_to_nodes(self.index) #Liste des noeuds de l'élément
        self.nodesCoord = {} # Dictionnaire Node : Coordonnées
        for node in self.nodes:
            self.nodesCoord[node] = np.array([mesh_obj.get_node_to_xcoord(node),mesh_obj.get_node_to_ycoord(node)])
        
        self.ElementCoord = np.zeros(2) #Coordonnées du centre de l'élément
        for coord in self.nodesCoord.values():
            self.ElementCoord += coord/len(self.nodesCoord)


############ CALCULS ################
    def ATA_add(self,Ald : np.ndarray):
        """
        Ajoute une matrice Ald à la matrice ATA de l'élément

        Parameters
        ----------
        Ald : ndarray
            Matrice à ajouter à la matrice ATA

        Returns
        -------
        None

        """
        self.ATA += Ald

    def B_add(self,index : int, Bld : np.ndarray):

        """
        Ajoute une matrice Bld à la matrice B de l'élément

        Parameters
        ----------
        index : int
            Index de la composante de la matrice B
        Bld : ndarray
            Matrice à ajouter à la matrice B

        Returns
        -------
        None

        """
        
        self.B[index] += Bld


    def calculGRAD(self):
        """
        Calcul du gradient de l'élément par produit matriciel, résolution

        Returns
        -------
        None

        """
        
        self.grad = np.linalg.inv(self.ATA)@self.B

    def calculFaceCenter(self, face : int):
        """
        Calcul du centre d'une face de l'élément

        Parameters
        ----------
        face : int
            Numéro de la face

        Returns
        -------
        ndarray
            Coordonnées du centre de la face
        """
        Nodes = self.mesh_obj.get_face_to_nodes(face)

        return self.nodesCoord[Nodes[0]] + (self.nodesCoord[Nodes[1]] - self.NodesCoord[Nodes[0]])/2  

    def calculFaceNormal(self, face : int):
        """
        Calcul de la normale d'une face de l'élément

        Parameters
        ----------
        face : int
            Numéro de la face

        Returns
        -------
        ndarray
            Composantes de la normale à la face
        """
        Nodes = self.mesh_obj.get_face_to_nodes(face)

        Direction = self.nodesCoord[Nodes[1]] - self.nodesCoord[Nodes[0]]
        Normal = np.array([Direction[1],-Direction[0]])
        Normal = Normal / np.linalg.norm(Normal)

        return Normal

    
########### GETTERS ################
    def get_Coord(self):
        return self.ElementCoord

    def get_value(self):
        return self.value
    
    def get_grad(self):
        return self.grad

########## SETTERS ################   
    def set_value(self, value):
        self.value = value


