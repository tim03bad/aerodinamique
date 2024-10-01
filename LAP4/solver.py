import numpy as np
import pyvista as pv
import pyvistaqt as pvQt
from sympy import true

from mesh import Mesh
import mesh
from meshPlotter import MeshPlotter

from utilities import Utilities
from element import Element
from meanSquare import MeanSquare
from dataFace import dataFace




class Solver:

    def __init__(self, mesh: Mesh,Lx : float,Ly : float,Gamma : float,q : float, CL : dict[int,(str,any)],crossDiffusion=True)-> None:

        # Initialisation du maillage
        self._mesh = mesh

        # Initialisation des constantes
        self._Gamma = Gamma
        self._q = q

        self._Lx = Lx
        self._Ly = Ly

        # Initialisation des conditions aux limites
        self._CL = CL

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

    def solve(self):
        for iter in range(1,5):


            #ReInitialisation des matrices
            self._A = np.zeros((len(self._PolyList),len(self._PolyList)))
            self._B = np.zeros(len(self._PolyList))

            #Terme source
            for i in range(len(self._PolyList)):
                self._B[i] += self._q*self._PolyList[i].get_Area()

            NbBoundaryFaces = self._mesh.get_number_of_boundary_faces()
            NbFaces = self._mesh.get_number_of_faces()

        
            print("\n\n Iteration : {}\n\n".format(iter))

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

                if self._CL[tag][0] == 'D':

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
                
                ##################  Pas sure du tout , j'approx (phi_b-phi_a)/dEta par gradPhi(A).eta
                    sdcross = -self._Gamma*(P_ksieta/P_nksi)*(Eg.get_grad()@eta)*DAi
                    
                    self._A[elems[0],elems[0]] += di
                    self._B[elems[0]] += sdcross + di*self._CL[tag][1]

                    
                    

                elif self._CL[tag][0] == 'N':
                    #print("Face : {}N, Phi={}".format(i,self._CL[tag][1]))
                    self._B[elems[0]] += self._Gamma*self._CL[tag][1]*DAi



            # Calcul hors fontières limites, uniquement face interne

            for i in range(NbBoundaryFaces,NbFaces):
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
                sdcross = self._util.Sdcross(P_nksi,P_ksieta,self._Gamma,DAi,Eg,Ed,eta)

                self._A[elems[0],elems[0]] += di
                self._A[elems[1],elems[1]] += di

                self._A[elems[0],elems[1]] -= di
                self._A[elems[1],elems[0]] -= di

                self._B[elems[0]] += sdcross
                self._B[elems[1]] += sdcross

            #Résolution matrice plein
            Phi = np.linalg.solve(self._A,self._B)

            #Mise à jour des éléments du maillage avec Phi
            for i in range(self._mesh.number_of_elements):
                self._PolyList[i].set_value(Phi[i])

            #Calcul du nouveau gradient du champs
            self._MS.calculMeanSquare()

    def plot(self):
        #Retrieving values for plotting
        Values = np.zeros(len(self._PolyList))

        for i in range(len(self._PolyList)):
            Values[i] = self._PolyList[i].get_value()

        plotter = MeshPlotter()
        nodes, elements = plotter.prepare_data_for_pyvista(self._mesh)

        pv_mesh = pv.PolyData(nodes, elements)
        pv_mesh['Champ T'] = Values  

        pl = pvQt.BackgroundPlotter()
        pl.add_mesh(pv_mesh, scalars='Champ T', show_edges=True, cmap='hot')

        pl.camera_position = 'xy'
        pl.show_grid()
        pl.show()

        #pause du programme
        self._util.pause()

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

    def coupeY(self,Y:float):
        
        for elem in self._PolyList:

            Nodes = self._mesh.get_element_to_nodes(elem.index)

            NodesCoords = np.array((len(Nodes),2))

            for node_i in range(len(Nodes)):
                NodesCoords[node_i,0] = self._mesh.get_node_to_xcoord(Nodes[node_i])
                NodesCoords[node_i,1] = self._mesh.get_node_to_ycoord(Nodes[node_i])

            print(NodesCoords)
            





                     



