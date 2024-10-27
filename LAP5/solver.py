#Mesh import
from unittest import result
from mesh import Mesh

#Python libraries
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


#Custom classes
from element import Element
from utilities import Utilities
from leastSquareSolver import LeastSquareSolver


class Solver():

    def __init__(self, mesh: Mesh, elements: list[Element],velocityField, CL: dict,rho:float, Gamma: float,S,pressureGrad,scheme: str = 'upwind',verbose: bool = False):

        self._verbose = verbose

        #Reference au maillage
        self._mesh = mesh

        #Réference aux éléments
        self._elements = elements

        #Conditions aux limites
        self._CL = CL

        #Terme source
        self._S = S
        self._pressureGrad = pressureGrad

        #Schemes
        self._upwind = True; #Default central scheme
        if scheme == 'central':
            self._upwind = False #Switch to central scheme
        




        #champs de vitesse
        self._velocityField = velocityField

        #Nombre de face et elements
        self._numberOfBoundaryFaces = self._mesh.get_number_of_boundary_faces()
        self._numberOfFaces = self._mesh.get_number_of_faces()
        self._numberOfElements = self._mesh.get_number_of_elements()

        #Coefficients
        self._Gamma = Gamma #Diffusivité thermique
        self._rho = rho #masse volumique

        #Matrices
        self._A = np.zeros((self._numberOfElements,self._numberOfElements))
        self._B = np.zeros(self._numberOfElements)


    def solve(self,iterations: int = 5):
        if self._verbose:
            print("Solving problem on {} elements ...".format(self._numberOfElements))
            if self._upwind:
                print("(Using upwind scheme)")
            else:
                print("(Using central scheme)")

        for i in range(1,iterations+1):
            if self._verbose:
                print("Solving iteration {} ...".format(i))

            #Reinitialisation des matrices
            
            self._A = np.zeros((self._numberOfElements,self._numberOfElements))
            self._B = np.zeros(self._numberOfElements)


            #Terme source
            for i in range(len(self._elements)):

                E = self._elements[i] #Element i


                if callable(self._S): #Si S est un champ scalaire
                    #Calcul de la moyenne de _S sur l'élément (Centre + noeud)
                    Smoy = 0

                    Ecenter = E._center
                    Smoy += self._S(Ecenter[0],Ecenter[1])

                    Nodes = self._mesh.get_element_to_nodes(i)
                    for node in Nodes:
                        CoordX = self._mesh.get_node_to_xcoord(node)
                        CoordY = self._mesh.get_node_to_ycoord(node)
                        Smoy += self._S(CoordX,CoordY)
                    
                    Smoy /=  (1+ len(Nodes))

                    self._B[i] += Smoy*E._area

                elif type(self._S) == np.ndarray:


                    self._B[i] += self._S[i]*E._area

                else:

                    self._B[i] += self._S*E._area


            #Boundary faces (conditions aux limites)
            for face_i in range(self._numberOfBoundaryFaces):

                tag = self._mesh.get_boundary_face_to_tag(face_i)
                elem = self._mesh.get_face_to_elements(face_i)
                Eg = self._elements[elem[0]]
                FrontierType = self._CL[tag][0]

                #Longueur et vecteurs
                N = Utilities.getFaceNormal(self._mesh, face_i)
                dAi = Utilities.getFaceLength(self._mesh, face_i)
                e_eta = Utilities.getFaceUnitVector(self._mesh, face_i)

                FaceCenter = Utilities.getFaceCenter(self._mesh, face_i)
                dKsi = np.linalg.norm(FaceCenter - Eg._center)
                e_ksi = (FaceCenter - Eg._center)/dKsi

                #Projections
                Pnksi = N@e_ksi
                Pksieta = e_ksi@e_eta

                #Flux convectif
                V = self._velocityField[Eg._index]

                F = (V[0]*N[0] + V[1]*N[1])*dAi

                 

                

                if FrontierType == 'D':
                    Phi = self._CL[tag][1]
                    if callable(Phi):
                        Phi = Phi(Eg._center[0], Eg._center[1])

                    #Terme de diffusion
                    Di = Solver.Di(Pnksi,self._Gamma,dAi,dKsi)
                    Sdi = Solver.Sdcross(Pnksi,Pksieta,self._Gamma,dAi,Eg._grad,Eg._grad,e_eta)

                    #Mise à jour des matrices | Diffusion
                    self._A[elem[0],elem[0]] += Di
                    self._B[elem[0]] += (Sdi + Phi*Di)

                    #Mise à jour des matrices | Convection
                    self._A[elem[0],elem[0]] += np.max([F,0])
                    self._B[elem[0]] += np.max([0,-F])*Phi
                

                elif FrontierType == 'N':

                    gradPhi = self._CL[tag][1]
                    if callable(gradPhi):
                        gradPhi = gradPhi(Eg._center[0], Eg._center[1])
                        gradPhiN = gradPhi@N

                    if type(gradPhi) == np.ndarray:
                        gradPhiN = gradPhi@N
                    else:
                        gradPhiN = gradPhi
                    
                    #mise à jour des matrices | Diffusion
                    self._B[elem[0]] += self._Gamma*dAi*gradPhiN

                    #Mise à jour des matrices | Convection
                    self._A[elem[0],elem[0]] += F
                    self._B[elem[0]] -= F*gradPhiN*Pnksi*dKsi

                    
                    

            

            #Internal faces
            for face_i in range(self._numberOfBoundaryFaces, self._numberOfFaces):
                #Préparation
                elem = self._mesh.get_face_to_elements(face_i)
                Eg = self._elements[elem[0]]
                Ed = self._elements[elem[1]]

                #Longueur et vecteurs
                dKsi = np.linalg.norm(Ed._center - Eg._center)
                dAi = Utilities.getFaceLength(self._mesh, face_i)
                N = Utilities.getFaceNormal(self._mesh, face_i)
                e_ksi = (Ed._center - Eg._center)/dKsi
                e_eta = Utilities.getFaceUnitVector(self._mesh, face_i)

                #Flux convectif
                FaceCenter = Utilities.getFaceCenter(self._mesh, face_i)

                Vg = self._velocityField[Eg._index]
                Vd = self._velocityField[Ed._index]
                Fxg = self._rho*Vg[0]
                Fyg = self._rho*Vg[1]
                Fxd = self._rho*Vd[0]
                Fyd = self._rho*Vd[1]

                Fx = (Fxg + Fxd)/2
                Fy = (Fyg + Fyd)/2

                F = (Fx*N[0] + Fy*N[1])*dAi

                #Projections
                Pnksi = N@e_ksi
                Pksieta = e_ksi@e_eta

                #Terme de diffusion
                Di = Solver.Di(Pnksi,self._Gamma,dAi,dKsi)
                Sdi = Solver.Sdcross(Pnksi,Pksieta,self._Gamma,dAi,Ed._grad,Ed._grad,e_eta)

                #Mise à jour des matrices | Diffusion
                self._A[elem[0],elem[0]] += Di
                self._A[elem[1],elem[1]] += Di
                self._A[elem[0],elem[1]] -= Di 
                self._A[elem[1],elem[0]] -= Di

                self._B[elem[0]] += Sdi
                self._B[elem[1]] -= Sdi

                #Mise à jour des matrices | Convection
                if self._upwind:
                    self._A[elem[0],elem[0]] += np.max([F,0]) #Tg,Tg
                    self._A[elem[1],elem[1]] += np.max([0,-F]) #Td,Td
                    self._A[elem[0],elem[1]] -= np.max([0,-F]) #Tg,Td
                    self._A[elem[1],elem[0]] -= np.max([F,0]) #Td,Tg

                else:
                    self._A[elem[0],elem[0]] += 0.5*F
                    self._A[elem[1],elem[1]] -= 0.5*F
                    self._A[elem[0],elem[1]] += 0.5*F
                    self._A[elem[1],elem[0]] -= 0.5*F
            

            #Solve
            
            if len(self._elements) <=500:
                self._solution = np.linalg.solve(self._A, self._B)
            else:
                if self._verbose:
                    print("Solving linear system with sparse matrix !")
                Asparse = self.convertToCsr(self._A)
                self._solution = spsolve(Asparse, self._B) 

            for elem in self._elements:
                elem._value = self._solution[elem._index]


            
            lss = LeastSquareSolver(self._mesh,self._elements,self._CL)
            lss.solve()
        if self._verbose:   
            print("Done!")
        return self._solution




    
    #Methodes
    @staticmethod
    def Di(Pnksi,k,dAi,dKsi):

        return (k*dAi)/(dKsi*Pnksi)
    
    @staticmethod
    def Sdcross(Pnksi: float,Pksieta: float,k :float,dAi : float,gradA : np.ndarray,gradP : np.ndarray,e_eta : np.ndarray):

        scal = ((gradA + gradP)/2)@e_eta

        return -k*(Pksieta*scal*dAi)/Pnksi

    @staticmethod
    def convertToCsr(A : np.ndarray):
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
    




