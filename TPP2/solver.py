
from matplotlib.testing import set_font_settings_for_testing
import numpy as np
import pyvista as pv
import pyvistaqt as pvQt



from mesh import Mesh

import mesh
from meshPlotter import MeshPlotter

from utilities import Utilities
from element import Element
from meanSquare import MeanSquare
from dataFace import dataFace

import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve




class Solver:

    def __init__(self, mesh: Mesh,Lx : float,Ly : float,Gamma : float,S,rho : float,Cp : float, CL : dict[int,(str,any)],velocityField,schema : str,crossDiffusion=True)-> None:

        # Initialisation du maillage
        self._mesh = mesh

        # Initialisation des constantes
        self._Gamma = Gamma
        self._S = S
        self._rho = rho
        self._Cp = Cp

        self._Lx = Lx
        self._Ly = Ly

        # Initialisation des conditions aux limites
        self._CL = CL

        # Initialisation du champ de vitesse
        if callable(velocityField) == True:
            self._velocityField = velocityField
        else:
            raise TypeError("velocityField must be a function")

        # Initialisation du schema de discretisation du terme convectif 
        self._upwind = False
        self._central = False
        if schema == 'upwind':
            self._upwind = True
        elif schema == 'central':
            self._central = True
        else:
            raise ValueError('schema must be \'upwind\' or \'central\'')

        # Initialisation des elements 
        self._PolyList = [Element(self._mesh, i) for i in range(self._mesh.get_number_of_elements())]

        # Initialisation des matrices
        self._A = np.zeros((len(self._PolyList),len(self._PolyList)))
        self._B = np.zeros(len(self._PolyList))

        # Initialisation solveur reconstruction Gradient
        self._MS = MeanSquare(self._mesh,self._PolyList,self._CL)

        # Initialisation utilities
        self._util = Utilities()

        self._crossDiffusion = crossDiffusion



    def solve(self,iteration : int = 1):

        for iter in range(1,iteration+1):


            #ReInitialisation des matrices
            self._A = np.zeros((len(self._PolyList),len(self._PolyList)))
            self._B = np.zeros(len(self._PolyList))

            #Terme source
            for i in range(len(self._PolyList)):

                E = self._PolyList[i] #Element i


                if callable(self._S): #Si S est un champ scalaire
                    #Calcul de la moyenne de _S sur l'élément (Centre + noeud)
                    Smoy = 0

                    Ecenter = E.get_Coord()
                    Smoy += self._S(Ecenter[0],Ecenter[1])

                    Nodes = self._mesh.get_element_to_nodes(i)
                    for node in Nodes:
                        NodeCoord = E.nodesCoord[node]
                        Smoy += self._S(NodeCoord[0],NodeCoord[1])
                    
                    Smoy /=  (1+ len(Nodes))

                    self._B[i] += Smoy*E.get_Area()

                else:
                    self._B[i] += self._S*E.get_Area()
                
                

            NbBoundaryFaces = self._mesh.get_number_of_boundary_faces()
            NbFaces = self._mesh.get_number_of_faces()

        
            print(" Iteration : {}".format(iter))

            #Face frontiere du domaine
            for i in range(NbBoundaryFaces):
                tag = self._mesh.get_boundary_face_to_tag(i)
                elems = self._mesh.get_face_to_elements(i)
                #Que le triangle gauche, le droit n'existe pas
                Eg = self._PolyList[elems[0]]

                FaceNodes = self._mesh.get_face_to_nodes(i)

                Xa = self._mesh.node_to_xcoord[FaceNodes[0]]
                Ya = self._mesh.node_to_ycoord[FaceNodes[0]]
                
                Xb = self._mesh.node_to_xcoord[FaceNodes[1]]
                Yb = self._mesh.node_to_ycoord[FaceNodes[1]]

                DAi = np.sqrt((Xb-Xa)**2+(Yb-Ya)**2)

                FaceCenter = Eg.calculFaceCenter(i)
                FaceNormal = Eg.calculFaceNormal(i)
                VelFace = self._velocityField(FaceCenter[0],FaceCenter[1]) #X,Y coordonnees du centre de la face
                F = self._rho*(FaceNormal@VelFace)*DAi

                if self._CL[tag][0] == 'D':
                    
                    ###### Gestion constance ou non de la condition au limite

                    if callable(self._CL[tag][1]):
                        fPhi = self._CL[tag][1] #fonction
                        PhiD = fPhi(FaceCenter[0],FaceCenter[1])
                    else:
                        PhiD = self._CL[tag][1]

                    ###### Diffusion

                    XP = Eg.ElementCoord[0]
                    YP = Eg.ElementCoord[1]

                    XA = (Xa+Xb)/2
                    YA = (Ya+Yb)/2

                    dKsi = np.sqrt((XA-XP)**2+(YA-YP)**2)

                    data = dataFace(Xa,Xb,Ya,Yb,XA,XP,YA,YP)

                    P_nksi = self._util.Pnksi(data,DAi,dKsi)
                    P_ksieta = self._util.Pksieta(data,dKsi,DAi)

                    eta = np.array([(data.Xb-data.Xa)/DAi,(data.Yb-data.Ya)/DAi])

                    di = (self._Gamma/P_nksi)*(DAi/dKsi)

                    #print("D, di = {:.3f}, dKsi = {:.3f}, DAi = {:.3f},Pnksi = {:.3f}, P_ksieta = {:.3f}".format(di,dKsi,DAi,P_nksi,P_ksieta))

    
                
                    #  Pas sure du tout , j'approx (phi_b-phi_a)/dEta par gradPhi(A).eta
                    sdcross = -self._Gamma*(P_ksieta/P_nksi)*(Eg.get_grad()@eta)*DAi
                    
                    self._A[elems[0],elems[0]] += di
                    self._B[elems[0]] += sdcross + di*PhiD


                    ################ Convection ################
                    #CL => upwind et central meme traitement
                    self._A[elems[0],elems[0]] += np.max([0,F])*self._Cp
                    self._B[elems[0]] += np.max([0,-F])*self._Cp*PhiD
                    
                    

                elif self._CL[tag][0] == 'N':

                    if callable(self._CL[tag][1]):
                        fGPhi = self._CL[tag][1] #gradient imposé
                        GPhi = fGPhi(FaceCenter[0],FaceCenter[1])
                        GPhiN = GPhi@FaceNormal
                    else:
                        GPhiN = self._CL[tag][1]

                    #print("Face : {}N, Phi={}".format(i,self._CL[tag][1]))
                    self._B[elems[0]] += self._Gamma*GPhiN*DAi
                    
                    ###### Convection ######

                    Eta = FaceCenter - Eg.ElementCoord
                    EtaN = Eta/np.linalg.norm(Eta)

                    #CL => upwind et central meme traitement
                    self._A[elems[0],elems[0]] += F*self._Cp
                    self._B[elems[0]] = self._B[elems[0]] + self._Gamma*GPhiN*DAi - F*GPhiN*(Eta@FaceNormal)*self._Cp


            #print("\n\n#####\n\nMatrice B av : {}".format(self._B))

            # Calcul hors fontières limites, uniquement face interne

            for i in range(NbBoundaryFaces,NbFaces):

                ###### Terme Diffusif ######

                elems = self._mesh.get_face_to_elements(i)
                #Elems[0] = Triangle gauche
                #Elems[1] = Triangle droit

                Eg = self._PolyList[elems[0]]
                Ed = self._PolyList[elems[1]]

                data = self._util.FaceCoordinatesCalcultion(self._mesh,i,self._PolyList)

                DAi = np.sqrt((data.Xa-data.Xb)**2+(data.Ya-data.Yb)**2)
                dKsi = np.sqrt((data.XA-data.XP)**2+(data.YA-data.YP)**2)

                P_nksi = self._util.Pnksi(data,DAi,dKsi)
                P_ksieta = self._util.Pksieta(data,dKsi,DAi)

                eta = np.array([data.Xb-data.Xa,data.Yb-data.Ya])

                di = self._util.Di(P_nksi,self._Gamma,DAi,dKsi)
                sdcross = self._util.Sdcross(P_nksi,P_ksieta,self._Gamma,DAi,Eg,Ed,eta,self._crossDiffusion)

                self._A[elems[0],elems[0]] += di
                self._A[elems[1],elems[1]] += di

                self._A[elems[0],elems[1]] -= di
                self._A[elems[1],elems[0]] -= di

                self._B[elems[0]] += sdcross
                self._B[elems[1]] -= sdcross

                ###### Terme Convectif ######

                FaceNormal = Eg.calculFaceNormal(i)
                FaceCenter = Eg.calculFaceCenter(i)
                VelFace = self._velocityField(FaceCenter[0],FaceCenter[1]) #X,Y coordonnees du centre de la face

                #Flux massique à travers la face i
                Fi = self._rho*(FaceNormal@VelFace)*DAi

                if self._central:

                    self._A[elems[0],elems[0]] += 0.5*Fi*self._Cp
                    self._A[elems[1],elems[1]] -= 0.5*Fi*self._Cp

                    self._A[elems[0],elems[1]] += 0.5*Fi*self._Cp
                    self._A[elems[1],elems[0]] -= 0.5*Fi*self._Cp

                elif self._upwind:

                    self._A[elems[0],elems[0]] += np.max([0,Fi])*self._Cp
                    self._A[elems[1],elems[1]] += np.max([0,-Fi])*self._Cp

                    self._A[elems[0],elems[1]] -= np.max([0,-Fi])*self._Cp
                    self._A[elems[1],elems[0]] -= np.max([0,Fi])*self._Cp

            #print("\n\n#####\n\nMatrice B aP : {}".format(self._B))
            
            #Résolution matrice plein
            if len(self._PolyList) <= 400:
                Phi = np.linalg.solve(self._A,self._B)
            else:
                
                print("Solving with sparse matrix")
                Asparse = self.convert_to_csr(self._A)
                Phi = spsolve(Asparse,self._B)

            #Mise à jour des éléments du maillage avec Phi
            for i in range(self._mesh.number_of_elements):
                self._PolyList[i].set_value(Phi[i])

            #Calcul du nouveau gradient du champs
            self._MS.calculMeanSquare()

    def plot(self,title:str,show_Mesh : bool = True,contour : bool = False):

        #Param d'affichage
        colorbar_args = {
            "title": "Champs T",  # Titre de la colorbar
            "vertical": True,  # Colorbar verticale
            "position_x": 0.05,  # Position horizontale de la colorbar (proche du bord gauche)
            "position_y": 0.1,   # Position verticale, centrée
            "width": 0.05,       # Largeur de la colorbar
            "height": 0.8        # Hauteur de la colorbar
        }


        #Retrieving values for plotting

        Values = np.zeros(len(self._PolyList))

        for i in range(len(self._PolyList)):
            Values[i] = self._PolyList[i].get_value()

        plotter = MeshPlotter()
        nodes, elements = plotter.prepare_data_for_pyvista(self._mesh)

        pv_mesh = pv.PolyData(nodes, elements)
        pv_mesh['Champ T'] = Values  

        pl = pvQt.BackgroundPlotter()
        pl.add_mesh(pv_mesh, scalars='Champ T', show_edges=show_Mesh, cmap='seismic',scalar_bar_args=colorbar_args)

        #Ajout des ligne de contour iso
        if contour:
            nodes_xcoords = self._mesh.get_nodes_to_xcoord()
            nodes_ycoords = self._mesh.get_nodes_to_ycoord()
            pv_mesh = pv_mesh.cell_data_to_point_data()
            contours = pv_mesh.contour(isosurfaces=15,scalars='Champ T')
            pl.add_mesh(contours,color='k',show_scalar_bar=False,line_width=2)


        pl.camera_position = 'xy'
        pl.add_text(title, position="upper_edge")
        pl.show()

    def errorQuadratique(self):
        
        error = 0
        for elem in self._PolyList:
            Coord = elem.get_Coord()
            error += elem.get_Area()*(self._SolAnalytique(Coord[0],Coord[1]) - elem.get_value())**2

        return np.sqrt(error/(self._Lx*self._Ly))
    
    def defineSolAnalytique(self,fct):

        if callable(fct):
            self._SolAnalytique = fct
        else:
            raise TypeError("SolAnalytique must be a function")
        
    def getMeanElementSize(self):
        return self._MS.calculTailleMoyenne()

    def coupeY(self,Y:float, plot : bool = True):
        
        Values = np.zeros(len(self._PolyList))

        for i in range(len(self._PolyList)):
            Values[i] = self._PolyList[i].get_value()

        plotter = MeshPlotter()
        nodes, elements = plotter.prepare_data_for_pyvista(self._mesh)

        pv_mesh = pv.PolyData(nodes, elements)
        pv_mesh['Champ T'] = Values 
        
        A = [pv_mesh.bounds[0],Y,0]
        B = [pv_mesh.bounds[1],Y,0]

        line_sample = pv_mesh.sample_over_line(A,B,resolution=100)

        values = line_sample['Champ T']
        points = line_sample.points

        distance = points[:,0]

        if plot:
            plt.plot(distance,values)
            plt.title("T(X) (coupe en Y = {})".format(Y))
            plt.grid()
            plt.xlim(pv_mesh.bounds[0],pv_mesh.bounds[1])
            plt.show()

        return distance,values

        
    
    def coupeX(self,X:float, plot : bool = True):
        Values = np.zeros(len(self._PolyList))

        for i in range(len(self._PolyList)):
            Values[i] = self._PolyList[i].get_value()

        plotter = MeshPlotter()
        nodes, elements = plotter.prepare_data_for_pyvista(self._mesh)

        pv_mesh = pv.PolyData(nodes, elements)
        pv_mesh['Champ T'] = Values 

        A = [X,pv_mesh.bounds[2],0]
        B = [X,pv_mesh.bounds[3],0]

        line_sample = pv_mesh.sample_over_line(A,B,resolution=100)
        
        values = line_sample['Champ T']
        points = line_sample.points

        distance = points[:,1]

        if plot:
            plt.plot(distance,values)
            plt.title("T(Y) (coupe en X = {})".format(X))
            plt.grid()
            plt.xlim(pv_mesh.bounds[0],pv_mesh.bounds[1])
            plt.show()

        return distance,values

    def getFieldValues(self):

        Values = np.zeros(len(self._PolyList))
        for i in range(len(self._PolyList)):
            Values[i] = self._PolyList[i].get_value()

        return Values


    @staticmethod
    def convert_to_csr(A : np.ndarray):
    # Initialiser les tableaux CSR
        data = []
        indices = []
        indptr = [0]  # Commence à 0, marque le début de la première ligne

        # Taille de la matrice
        n_rows, n_cols = A.shape

        # Parcourir chaque ligne de la matrice
        for i in range(n_rows):
            for j in range(n_cols):
                if A[i, j] != 0:  # Si la valeur n'est pas nulle
                    data.append(A[i, j])   # Ajouter la valeur non nulle à 'data'
                    indices.append(j)      # Ajouter l'indice de la colonne à 'indices'
            indptr.append(len(data))  # Ajouter la position actuelle dans 'data'

        # Convertir les listes en tableaux numpy
        data = np.array(data)
        indices = np.array(indices)
        indptr = np.array(indptr)

        return csr_matrix((data, indices, indptr), shape=A.shape)


            





                     



