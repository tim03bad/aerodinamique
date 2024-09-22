''' 
############   LAP 3    ###############
Emilien VIGIER
Timothée BADICHE
#############           ###############
'''

#%% Importations
import numpy as np
import mesh
from meshGenerator import MeshGenerator
from meshConnectivity import MeshConnectivity

#%% Création du maillage
mesher = MeshGenerator()

print('Rectangle : maillage non structuré avec des triangles')
mesh_parameters = {'mesh_type': 'TRI','lc': 0.05}
mesh_obj = mesher.rectangle([0.0, 1.0, 0.0, 1.0], mesh_parameters)
conec = MeshConnectivity(mesh_obj)
conec.compute_connectivity()

#%% Récupération des données du maillage

'''récupère le centre d'un élément'''
def get_center_element(mesh_obj, i_element):
    nodes = mesh_obj.get_element_to_nodes(i_element)
    x, y = 0, 0
    for node in nodes:
        coord = mesh_obj.get_node_to_xycoord(node)
        x += coord[0]
        y += coord[1]
    return (x/3, y/3)

'''récupère le centre d'une face'''
def get_center_face(mesh_obj, i_face):
    nodes = mesh_obj.get_face_to_nodes(i_face)
    x, y = 0, 0
    for node in nodes:
        coord = mesh_obj.get_node_to_xycoord(node)
        x += coord[0]
        y += coord[1]
    return (x/2, y/2)

'''récupère les voisins d'un élément'''
def get_neighbors(mesh_obj, i_element):
    Voisins = []
    num_face = []
    Liste_faces = mesh_obj.get_faces_to_elements()
    for nb_face in range(len(Liste_faces)):
        face = Liste_faces[nb_face]
        if i_element in face:
            num_face.append(nb_face)
            voisin = (face[face != i_element][0])
            Voisins.append(voisin)
    return (Voisins, num_face)

'''calcul la normal d'une face'''
def normal(mesh_obj, i_element, i_face):
    (xa,ya) = get_center_face(mesh_obj, i_face)
    (xtg,ytg) = get_center_element(mesh_obj, i_element)
    nodes = mesh_obj.get_face_to_nodes(i_face)
    
    X = np.zeros(2)
    X[0] = xa - xtg
    X[1] = ya - ytg
    
    vect_frontiere = np.zeros(2)
    vect_frontiere[0] = mesh_obj.get_node_to_xcoord(nodes[0]) - mesh_obj.get_node_to_xcoord(nodes[1])
    vect_frontiere[1] = mesh_obj.get_node_to_ycoord(nodes[0]) - mesh_obj.get_node_to_ycoord(nodes[1])
    
    nx = vect_frontiere[1]
    ny = -vect_frontiere[0]
    norme = np.sqrt(nx**2 + ny**2)
    nx = nx/norme
    ny = ny/norme
    
    if np.dot(np.array([nx, ny]),X)<0:
        nx = -nx
        ny = -ny
        
    return (nx, ny)

'''exemple de phi'''
def phi(x,y):
    return 2*x + 3*y
    
#%% Création de la matrice A sans frontière
def ALS(mesh_obj,i_element,voisins):
    (x0, y0) = get_center_element(mesh_obj, i_element)
    voisins = get_neighbors(mesh_obj, i_element)[0]
    (x1, y1) = get_center_element(mesh_obj, voisins[0])
    (x2, y2) = get_center_element(mesh_obj, voisins[1])
    (x3, y3) = get_center_element(mesh_obj, voisins[2])
    
    deltaX = np.zeros(3)
    deltaX[0] = x1-x0
    deltaX[1] = x2-x0
    deltaX[2] = x3-x0
    
    deltaY = np.zeros(3)
    deltaY[0] = y1-y0
    deltaY[1] = y2-y0
    deltaY[2] = y3-y0
    
    ALS = np.zeros((2,2))
    ALS[0][0] = np.dot(deltaX,deltaX)
    ALS[0][1] = np.dot(deltaX,deltaY)
    ALS[1][0] = np.dot(deltaX,deltaY)
    ALS[1][1] = np.dot(deltaX,deltaX)
    return ALS
    
#%% Création de la matrice ALS avec frontière
def ALS_front(mesh_obj,i_element,i_face,voisins,newmann=False):
    (xa,ya) = get_center_face(mesh_obj, i_face)
    (xtg,ytg) = get_center_element(mesh_obj, i_element)
    (x1, y1) = get_center_element(mesh_obj, voisins[0])
    (x2, y2) = get_center_element(mesh_obj, voisins[1])
    ALS_front = np.zeros((2,2))
    ALS_front[0][0] = (x1-xtg)**2
    ALS_front[0][1] = (x1-xtg)*(x2-xtg)
    ALS_front[1][0] = (x1-xtg)*(x2-xtg)
    ALS_front[1][1] = (x2-xtg)**2
    
    
    deltaX = xa-xtg
    deltaY = ya-ytg
    (nx, ny) = normal(mesh_obj, i_element, i_face)
    
    if newmann:
        deltaX = (deltaX*nx+deltaY*ny)*nx
        deltaY = (deltaX*nx+deltaY*ny)*ny
    
    ALS_front = np.zeros((2,2))
    ALS_front[0][0] += deltaX**2
    ALS_front[0][1] += deltaX*deltaY
    ALS_front[1][0] += deltaX*deltaY
    ALS_front[1][1] += deltaY**2
    return ALS_front

#%% Création de la matrice B=AT*Phi
def B_inter(mesh_obj,i_element,voisins):
    (x0, y0) = get_center_element(mesh_obj, i_element)
    (x1, y1) = get_center_element(mesh_obj, voisins[0])
    (x2, y2) = get_center_element(mesh_obj, voisins[1])
    (x3, y3) = get_center_element(mesh_obj, voisins[2])
    
    phi0 = phi(x0,y0)
    phi1 = phi(x1,y1)
    phi2 = phi(x2,y2)
    phi3 = phi(x3,y3)
    
    B = np.zeros(2)
    B[0] = (x1-x0)*(phi1-phi0) + (x2-x0)*(phi2-phi0) + (x3-x0)*(phi3-phi0)
    B[1] = (y1-y0)*(phi1-phi0) + (y2-y0)*(phi2-phi0) + (y3-y0)*(phi3-phi0)
    return B
    
#%% Création de la matrice B=AT*Phi à la frontière
def B_inter_front(mesh_obj, i_element, i_face,voisins,newmann=False):
    (xtg,ytg) = get_center_element(mesh_obj, i_element)
    (xa,ya) = get_center_face(mesh_obj, i_face)
    (x1, y1) = get_center_element(mesh_obj, voisins[0])
    (x2, y2) = get_center_element(mesh_obj, voisins[1])
    
    phitg = phi(xtg,ytg)
    phi1 = phi(x1,y1)
    phi2 = phi(x2,y2)
    B = np.zeros(2)
    
    B[0] = (x1-xtg)*(phi1-phitg) + (x2-xtg)*(phi2-phitg)
    B[1] = (y1-ytg)*(phi1-phitg) + (y2-ytg)*(phi2-phitg)
    
    if newmann:
        nodes = mesh_obj.get_face_to_nodes(i_face)
        vect_frontiere = np.zeros(2)
        vect_frontiere[0] = mesh_obj.get_node_to_xcoord(nodes[0]) - mesh_obj.get_node_to_xcoord(nodes[1])
        vect_frontiere[1] = mesh_obj.get_node_to_ycoord(nodes[0]) - mesh_obj.get_node_to_ycoord(nodes[1])
        
        n = np.array(normal(mesh_obj, i_element, i_face))
        projection = np.dot(vect_frontiere, n)
        
        xan = n[0]*projection
        yan = n[1]*projection
        phian = phi(xan,yan)
        
        B[0] += (xan-xtg)*(phian-phitg)
        B[1] += (yan-ytg)*(phian-phitg)
        return B
    
    phia = phi(xa,ya)

    B[0] += (xa-xtg)*(phia-phitg)
    B[1] += (ya-ytg)*(phia-phitg)
    return B
    
#%% matrice de matrice inversée ATAI
def ATAI(mesh_obj):
    nb_obj = mesh_obj.get_number_of_elements()
    ATA = np.zeros((nb_obj,2,2))
    for i in range(nb_obj):
        (voisins, num_face) = get_neighbors(mesh_obj, i)
        if -1 in voisins:
            index = voisins.index(-1)
            frontiere = num_face[index]
            voisins.remove(-1)
            ATA[i] = np.linalg.inv(ALS_front(mesh_obj,i,frontiere, voisins))
        else :
            ATA[i] = np.linalg.inv(ALS(mesh_obj,i,voisins))
    return ATA
            
#%% matrice de matrice B 
def B_tot(mesh_obj):
    nb_obj = mesh_obj.get_number_of_elements()
    B = np.zeros((nb_obj,2))
    for i in range(nb_obj):
        (voisins, num_face) = get_neighbors(mesh_obj, i)
        if -1 in voisins:
            index = voisins.index(-1)
            frontiere = num_face[index]
            voisins.remove(-1)
            B[i] = B_inter_front(mesh_obj, i, frontiere, voisins)
        else :
            B[i] = B_inter(mesh_obj,i,voisins)
    return B
    
#%% Calcul du gradient (GRAD = ATAI*B)
def Grad(mesh_obj):
    nb_faces = mesh_obj.get_number_of_elements()
    grad = np.zeros((nb_faces,2))
    ATA = ATAI(mesh_obj)
    B = B_tot(mesh_obj)
    for tri in range(nb_faces):
        grad[tri] = np.dot(ATA[tri],B[tri])
    return grad

print(Grad(mesh_obj))