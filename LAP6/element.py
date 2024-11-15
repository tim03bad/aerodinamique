import numpy as np

from mesh import Mesh

class Element:
    """
    Classe Element: stockage des données d'un élément
    MEC6616 - Aérodynamique numérique
    Date de création: 2021 - 12 - 22
    Auteur: El Hadji Abdou Aziz NDIAYE
    """

    def __init__(self,mesh: Mesh, index: int):

        #Reference au maillage
        self._mesh = mesh

        #Index de l'élément
        self._index = index

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

        #Champs de pression
        self._P = 0
        #Initialisation du gradient de pression à 0
        self._gradP = np.zeros(2)

        #Initialisation du champs de vitesse
        self._V = np.zeros(2)

        #Initialisation des gradients de vitesse
        self._gradU = np.zeros(2)
        self._gradV = np.zeros(2)

        

    def getCenter(self)->np.ndarray:
        return self._center
    
    def setPressure(self, P):
        if callable(P):
            self._P = P(self._center[0],self._center[1])
        else:
            self._P = P

    def setPressureGradient(self, gradP):
        if callable(gradP):
            self._gradP = gradP(self._center[0],self._center[1])
        else:
            self._gradP = gradP
    

