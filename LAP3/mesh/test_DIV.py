import unittest


from mesh import Mesh
import meshGenerator
from meshConnectivity import MeshConnectivity
from meshPlotter import MeshPlotter
import sympy as sp
import numpy as np
import pyvista as pv
import pyvistaqt as pvQt

from test import DivElements, NodesCoord


class TestMesh(unittest.TestCase):

    def test_mesh(self):

        V = np.array([1,1]) # Champs vectoriel constant
        


        mesher = meshGenerator.MeshGenerator()
        mesh_parameters = {'mesh_type': 'TRI','lc': 0.5}
        mesh_obj = mesher.rectangle([0.0, 1.0, 0.0, 1.0], mesh_parameters)

        conec = MeshConnectivity(mesh_obj)
        conec.compute_connectivity()

        V = np.array([1,1]) # Champs vectoriel constant

        DivElements = np.zeros(mesh_obj.get_number_of_elements())

        for i_face in range(mesh_obj.get_number_of_faces()):

            #On récupère les noeuds de l'arrête i
            Nodes = mesh_obj.get_face_to_nodes(i_face)


            #On récupère les coordonnées des noeuds
            NodesCoord = np.zeros((2,2))
            NodesCoord[0]= [mesh_obj.get_node_to_xcoord(Nodes[0]),mesh_obj.get_node_to_ycoord(Nodes[0])]
            NodesCoord[1]= [mesh_obj.get_node_to_xcoord(Nodes[1]),mesh_obj.get_node_to_ycoord(Nodes[1])]
            
            #On recupère les élements
            Elements = mesh_obj.get_face_to_elements(i_face)

            #Calcul du vecteur direction
            Direction = NodesCoord[1]-NodesCoord[0]

            #Calcul de la normal
            Normal = np.array([Direction[1],-Direction[0]])
            Normal = Normal / np.linalg.norm(Normal)

            #Calcul du flux
            F = V@Normal*np.linalg.norm(Direction)

            #Mise à jour divergence
            DivElements[Elements[0]] = DivElements[Elements[0]] + F
            if Elements[1] != -1: #Si on est sur une face externe, l'élements droit, non existant est indiqué par -1
                DivElements[Elements[1]] = DivElements[Elements[1]] - F

        #Application d'un seuil a 10^-12 au erreur de flottant
        DivElements[np.abs(DivElements) < 1e-10] = 0
        print(DivElements)

        #On verifie que la divergence est nulle sur tous les elements
        self.assertTrue(np.sum(DivElements)==0)

if __name__ == '__main__' :
    unittest.main()

