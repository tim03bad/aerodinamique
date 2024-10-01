#%%
from matplotlib.patheffects import Normal
import numpy as np
from meshGenerator import MeshGenerator
from meshConnectivity import MeshConnectivity
from mesh import Mesh

import pyvista as pv
import pyvistaqt as pvQt
from meshPlotter import MeshPlotter
#%%

def champ(x,y):
    return np.sin(x) + np.cos(y)

def grad(x,y):
    return np.array([np.cos(x),-np.sin(y)])
#%%

def ElementCenter(mesh_obj : Mesh, i_element):

    nodes = mesh_obj.get_element_to_nodes(i_element)
    Coord = np.zeros((len(nodes),2))

    for i in range(len(nodes)):
        Coord[i] = [mesh_obj.get_node_to_xcoord(nodes[i]),mesh_obj.get_node_to_ycoord(nodes[i])]
    
    nodes_closed = np.vstack([Coord, Coord[0]])
        
        #Aire avec la formule du determinant
    A = 0.5 * np.sum(nodes_closed[:-1, 0] * nodes_closed[1:, 1] - nodes_closed[1:, 0] * nodes_closed[:-1, 1])

    C_x = np.sum((nodes_closed[:-1,0] + nodes_closed[1:,0]) * 
                     (nodes_closed[:-1,0] * nodes_closed[1:,1] - nodes_closed[1:,0]*nodes_closed[:-1,1])) / (6*A)

    C_y = np.sum((nodes_closed[:-1, 1] + nodes_closed[1:, 1]) * 
                 (nodes_closed[:-1, 0] * nodes_closed[1:, 1] - nodes_closed[1:, 0] * nodes_closed[:-1, 1])) / (6 * A)

    return np.array([C_x,C_y])

def faceCenter(mesh_obj : Mesh, i_face):

    Nodes = mesh_obj.get_face_to_nodes(i_face)

    NodesCoord = np.zeros((len(Nodes),2))
    for i in range(len(Nodes)):
        NodesCoord[i] = [mesh_obj.get_node_to_xcoord(Nodes[i]),mesh_obj.get_node_to_ycoord(Nodes[i])]

    return (NodesCoord[0] + NodesCoord[1])/2


def meanArea(mesh_obj : Mesh):
    listOfArea = []
    for i in range(mesh_obj.get_number_of_elements()):
        nodes = mesh_obj.get_element_to_nodes(i)
        vertices = np.array([[mesh_obj.get_node_to_xcoord(node),mesh_obj.get_node_to_ycoord(node)] for node in nodes])
        vertices = np.vstack([vertices, vertices[0]])

        #Calcul de l'aire de l'élément avec la formule du shoelace
        listOfArea.append(0.5 * np.abs(np.sum(vertices[:-1,0] * 
                                              vertices[1:,1] - vertices[1:,0]*vertices[:-1,1])))
        
    listOfArea = np.array(listOfArea)
    print("Surface Total : ",np.sum(listOfArea))

    return (np.sqrt(np.sum(listOfArea**2)/len(listOfArea)) , listOfArea)

#%%
mesher = MeshGenerator(verbose=False)
plotter = MeshPlotter()


#Creation du maillage
mesh_parameters = {'mesh_type': 'QUAD', 'Nx': 200, 'Ny': 200}
mesh_obj = mesher.rectangle([0.0, 1.0, 0.0, 1.0], mesh_parameters)
conec = MeshConnectivity(mesh_obj)  
conec.compute_connectivity()

#plotter.plot_mesh(mesh_obj, label_points=True, label_elements=True, label_faces=True)

#%%
#Creation élements

number_of_elements = mesh_obj.get_number_of_elements()

ElementCoord = np.array([ElementCenter(mesh_obj,i) for i in range(number_of_elements)])
ElementValue = np.array([champ(ElementCoord[i][0],ElementCoord[i][1]) for i in range(number_of_elements)])

Gradient = np.zeros((number_of_elements,2))

ATA = np.zeros((number_of_elements,2,2))
B = np.zeros((number_of_elements,2))

#Condition aux limites (ici dirichle)
bcdata = (['D',champ],['D',champ],['D',champ],['D',champ])

#Boundary faces

for i in range(mesh_obj.get_number_of_boundary_faces()):
    tag = mesh_obj.get_boundary_face_to_tag(i)
    bc_type = bcdata[tag][0]
    bc_value = bcdata[tag][1]

    if ( bc_type == 'D' ):
        Eg = mesh_obj.get_face_to_elements(i)[0]

        #ATA 
        FaceCenter = faceCenter(mesh_obj, i)
        EgCenter = ElementCoord[Eg]


        DX = FaceCenter[0] - EgCenter[0]
        DY = FaceCenter[1] - EgCenter[1]

        A = np.zeros((2,2))

        A[0,0] = DX**2
        A[1,0] = DX*DY
        A[0,1] = DX*DY
        A[1,1] = DY**2

        ATA[Eg] += A
        

        #B

        FaceValue = champ(FaceCenter[0],FaceCenter[1])
        Bt = np.zeros(2)
        Bt[0] = DX*(FaceValue-ElementValue[Eg])
        Bt[1] = DY*(FaceValue-ElementValue[Eg])
        B[Eg] += Bt




for i_face in range(mesh_obj.get_number_of_boundary_faces(),mesh_obj.get_number_of_faces()):


    elements = mesh_obj.get_face_to_elements(i_face)
    Eg = elements[0]
    Ed = elements[1]


    EgCenter = ElementCoord[Eg]
    EdCenter = ElementCoord[Ed]

    DX = EdCenter[0] - EgCenter[0]
    DY = EdCenter[1] - EgCenter[1]

    #ATA

    A = np.zeros((2,2))

    A[0,0] = DX**2
    A[1,0] = DX*DY
    A[0,1] = DX*DY
    A[1,1] = DY**2


    ATA[Eg] += A
    ATA[Ed] += A

    #B
    EgValue = ElementValue[Eg]
    EdValue = ElementValue[Ed]

    Bt = np.zeros(2)
    Bt[0] = DX*(EdValue-EgValue)
    Bt[1] = DY*(EdValue-EgValue)
    B[Eg] += Bt
    B[Ed] += Bt

for i_elem in range(number_of_elements):
    Gradient[i_elem] = np.linalg.inv(ATA[i_elem])@B[i_elem]

GradA =np.array([grad(ElementCoord[i][0],ElementCoord[i][1]) for i in range(number_of_elements)])

print("Ei : Analytique | Numerique")
#for i_elem in range(number_of_elements):
    #print("E{} : {} | {}".format(i_elem,GradA[i_elem],Gradient[i_elem]))

#%% Erreur 1

h1,listOfArea = meanArea(mesh_obj)

NormA = np.array([np.linalg.norm(GradA[i]) for i in range(number_of_elements)])
NormN = np.array([np.linalg.norm(Gradient[i]) for i in range(number_of_elements)])

E1 = np.sqrt(np.sum(listOfArea*((NormA-NormN)**2)))
print("Erreur 1 : ",E1)
print("Taille moyenne :",h1)

#%% Erreur 2
h2,listOfArea = meanArea(mesh_obj)
NormA = np.array([np.linalg.norm(GradA[i]) for i in range(number_of_elements)])
NormN = np.array([np.linalg.norm(Gradient[i]) for i in range(number_of_elements)])

E2 = np.sqrt(np.sum(listOfArea*((NormA-NormN)**2)))
    
print("Erreur 2 : ",E2)
print("Taille moyenne :",h2)
        





# %%
