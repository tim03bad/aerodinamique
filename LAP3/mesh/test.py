
#%%
from altair import Element
from matplotlib.patheffects import Normal
import pyvista as pv
import pyvistaqt as pvqt
import numpy as np
from sympy import Nor

from mesh import Mesh
from meshGenerator import MeshGenerator
from meshConnectivity import MeshConnectivity
from meshPlotter import MeshPlotter

#%%
mesher = MeshGenerator()
plotter = MeshPlotter()

#%%
mesh_parameters = {'mesh_type': 'TRI','lc': 0.5}
mesh_obj = mesher.rectangle([0.0, 1.0, 0.0, 1.0], mesh_parameters)
conec = MeshConnectivity(mesh_obj)
conec.compute_connectivity()


#%%
element = Mesh.get_number_of_elements(mesh_obj) 
noeud = Mesh.get_number_of_nodes(mesh_obj)
face = Mesh.get_number_of_faces(mesh_obj)

print(f"Nombre de face {face}")
print(f"Nombre de noeud {noeud}")
print(f"Nombre d'element {element}")


# %%
V = np.array([1,1]) # Champs vectoriel constant

#%%
face_i = 0
Nodes = mesh_obj.get_face_to_nodes(face_i)
NodesCoord = np.zeros((2,2))
Elements= mesh_obj.get_face_to_elements(face_i)
NodesCoord[0]= [mesh_obj.get_node_to_xcoord(Nodes[0]),mesh_obj.get_node_to_ycoord(Nodes[0])]
NodesCoord[1]= [mesh_obj.get_node_to_xcoord(Nodes[1]),mesh_obj.get_node_to_ycoord(Nodes[1])]
Direction = NodesCoord[1]-NodesCoord[0]
Normal = np.array([Direction[1],-Direction[0]])
Normal = Normal / np.linalg.norm(Normal)

F = V@Normal*np.linalg.norm(Direction)

print(F)



# %%

DivElements = np.zeros(mesh_obj.get_number_of_elements())
#%%
# On parcourt tout les face/arrêtes les unes après les autres
for i_face in range(Mesh.get_number_of_faces(mesh_obj)):

    #On récupère les noeuds de l'arrête i
    Nodes = mesh_obj.get_face_to_nodes(i_face)

    #On récupère les coordonnées des noeuds
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

    #
    DivElements[Elements[0]] = DivElements[Elements[0]] + F
    if Elements[1] != -1: #Si on est sur une face externe, l'élements droit, non existant est indiqué par -1
        DivElements[Elements[1]] = DivElements[Elements[1]] - F

#Application d'un seuil a 10^-12 au erreur de flottant
DivElements[np.abs(DivElements) < 1e-10] = 0
print(DivElements)

np.sum(DivElements)==0


# %%
