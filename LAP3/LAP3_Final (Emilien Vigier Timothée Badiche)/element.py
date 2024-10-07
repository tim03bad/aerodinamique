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

        Notes
        -----
        La classe Element est utilis e pour stocker les informations
        n cessaires pour l'assemblage du syst me de l' quation
        aux d riv es partielles. Les grandeurs qui sont stock es
        dans cet objet sont :
        
        * La valeur du champ au centre de l' élément
        * Le gradient du champ au centre de l'élément
        * La matrice ATA de l'élément
        * La matrice B de l'élément
        * Les coordonnées du centre de l'élément
        * Les noeuds de l'élément
        * Les coordonnées des noeuds de l'élément

        """

        self.index = index
        self.mesh_obj = mesh_obj #Réference au maillage

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
        
        #Calcul du centre de l'élément
        self.ElementCoord = self.__private_Center()

        #Calcul de l'aire de l'élément
        self.area = self.__private_Area()

        ####Debug#####
        self.Blist = []
        self.Alist = []
        self.AListT = []


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

    def B_add(self,Bld : np.ndarray):

        """
        Ajoute une matrice Bld à la matrice B de l'élément

        Parameters
        ----------
        Bld : ndarray
            Matrice à ajouter à la matrice B

        Returns
        -------
        None

        """
        
        self.B += Bld


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

        return (self.nodesCoord[Nodes[0]] + self.nodesCoord[Nodes[1]])/2

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
    
############ PRIVATE ################

    def __private_Area(self):

        n = len(self.nodes)
        vertices = np.array([self.nodesCoord[node] for node in self.nodes])
        vertices = np.vstack([vertices, self.nodesCoord[self.nodes[0]]])

        #Calcul de l'aire de l'élément avec la formule du shoelace
        return 0.5 * np.abs(np.sum(vertices[:-1,0]* vertices[1:,1] - vertices[1:,0]*vertices[:-1,1]))


    
    def __private_Center(self):

        # #construction du vecteur des noeuds
        nodes = np.zeros((len(self.nodes),2))

        for i in range(len(self.nodes)):
            nodes[i] = self.nodesCoord[self.nodes[i]]
        
        # nodes_closed = np.vstack([nodes, nodes[0]])
        
        # #Aire avec la formule du determinant
        # A = 0.5 * np.sum(nodes_closed[:-1, 0] * nodes_closed[1:, 1] - nodes_closed[1:, 0] * nodes_closed[:-1, 1])

        # C_x = np.sum((nodes_closed[:-1,0] + nodes_closed[1:,0]) * 
        #              (nodes_closed[:-1,0] * nodes_closed[1:,1] - nodes_closed[1:,0]*nodes_closed[:-1,1])) / (6*A)

        # C_y = np.sum((nodes_closed[:-1, 1] + nodes_closed[1:, 1]) * 
        #          (nodes_closed[:-1, 0] * nodes_closed[1:, 1] - nodes_closed[1:, 0] * nodes_closed[:-1, 1])) / (6 * A)
        
        # return np.array([C_x,C_y])

        return np.mean(nodes, axis=0) #Centre de gravité de l'élément

    
########### GETTERS ################
    def get_Coord(self):
        return self.ElementCoord

    def get_value(self):
        return self.value
    
    def get_grad(self):
        return self.grad
    
    def get_Area(self):
        return self.area
    
    def getGradNorm(self):
        return np.linalg.norm(self.grad)

########## SETTERS ################   
    def set_value(self, value):
        self.value = value


########## DEBUG ################
    def storeB(self,Bld : np.ndarray):
        self.Blist.append(Bld)

    def storeA(self,Ald : np.ndarray,T : tuple):
        self.Alist.append(Ald)
        self.AListT.append(T)

