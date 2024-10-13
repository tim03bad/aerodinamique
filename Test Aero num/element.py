import numpy as np

from mesh import Mesh

class Element:
    """
    Classe Element: stockage des données d'un élément
    MEC6616 - Aérodynamique numérique
    Date de création: 2021 - 12 - 22
    Auteur: El Hadji Abdou Aziz NDIAYE
    """

    def __init__(self,mesh: Mesh, index: int,value):

        #Reference au maillage
        self._mesh = mesh

        #Index de l'élément
        self._index = index

        #Matrice de l'émélement
        self._A = np.zeros((2, 2))
        self._B = np.zeros(2)

        #Nodes de l'élément
        nodes = self._mesh.get_element_to_nodes(self._index)
        self._nodes = np.zeros((len(nodes), 2))
    
        for i in range(len(nodes)):
            self._nodes[i,0]=self._mesh.get_node_to_xcoord(nodes[i]) #X coord
            self._nodes[i,1]=self._mesh.get_node_to_ycoord(nodes[i]) #Y coord

        #Centre de l'élement
        self._center = np.mean(self._nodes, axis=0)

        #Aire de l'élement
        self._area = 0
        for i in range(len(self._nodes)):
            j = (i+1)%len(self._nodes)
            self._area += self._nodes[i,0]*self._nodes[j,1] - self._nodes[j,0]*self._nodes[i,1]
        self._area = abs(self._area)/2

        #Valeur
        if callable(value):
            self._value = value(self._center[0],self._center[1])
        else:
            self._value = value
        #Initialisation du gradient à 0
        self._grad = np.zeros(2)



    
    def addA(self, A):
        self._A += A

    def addB(self, B):
        self._B += B

    def solve(self):
        self._grad = np.linalg.solve(self._A, self._B)

    def getCenter(self)->np.ndarray:
        return self._center
