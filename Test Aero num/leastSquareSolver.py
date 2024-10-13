#Mesh import
from mesh import Mesh

#Python libraries
import numpy as np
import pyvista as pv

#Custom classes
from element import Element
from utilities import Utilities


class LeastSquareSolver:

    def __init__(self, mesh: Mesh,elements: list[Element],CL: dict, verbose: bool = False):

        self._verbose = verbose

        #Reference au maillage
        self._mesh = mesh

        #Réference aux éléments
        self._elements = elements

        #Conditions aux limites
        self._CL = CL

        #Nombre de face 
        self._numberOfBoundaryFaces = self._mesh.get_number_of_boundary_faces()
        self._numberOfFaces = self._mesh.get_number_of_faces()

    
    def solve(self):

        if self._verbose:
            print("Solving least square problem on {} elements ...".format(len(self._elements)))
        
        #Construction des matrices ATA et B pour les faces frontières
        self.__private__buildingBoundaries()
        self.__private__buildingInternal()
            
            
        for elem in self._elements:
            elem.solve()
        if self._verbose:
            print("Done!")

    def __private__buildingBoundaries(self):

        for face_i in range(self._numberOfBoundaryFaces):

            #Préparation
            tag = self._mesh.get_boundary_face_to_tag(face_i)
            elem = self._mesh.get_face_to_elements(face_i)
            Eg = self._elements[elem[0]]
            FrontierType = self._CL[tag][0]

            FaceCenter = Utilities.getFaceCenter(self._mesh, face_i)
            FaceNormal = Utilities.getFaceNormal(self._mesh, face_i)

            dX = FaceCenter[0] - Eg._center[0]
            dY = FaceCenter[1] - Eg._center[1]

            #Matrice ATA temporaire
            Als = np.zeros((2,2))
            Bls = np.zeros(2)   


            if FrontierType == 'D':

                Phi = self._CL[tag][1]
                if callable(Phi):
                    Phi = Phi(FaceCenter[0], FaceCenter[1])

                #Matrice A ############
                
                Als[0,0] = dX**2
                Als[1,0] = dX*dY
                Als[0,1] = dY*dX
                Als[1,1] = dY**2
                

                #Matrice B ############

                Bls[0] = dX*(Phi-Eg._value)
                Bls[1] = dX*(Phi-Eg._value)


            elif FrontierType == 'N':
                
                gradPhi = self._CL[tag][1]
                if callable(gradPhi):
                    gradPhi = gradPhi(FaceCenter[0], FaceCenter[1])
                    gradPhiN = gradPhi@FaceNormal
                elif type(gradPhi) == np.ndarray:
                    gradPhiN = gradPhi@FaceNormal
                else: #gradPhi is a scalar, normal gradient
                    gradPhiN = gradPhi
        

                dXn = (dX*FaceNormal[0] + dY*FaceNormal[1])*FaceNormal[0]
                dYn = (dX*FaceNormal[0] + dY*FaceNormal[1])*FaceNormal[1]
                dPhi = (dX*FaceNormal[0] + dY*FaceNormal[1])*gradPhiN


                #Matrice A   ##############
                Als[0,0] = dXn**2
                Als[1,0] = dXn*dYn
                Als[0,1] = dYn*dXn
                Als[1,1] = dYn**2

                #Matrice B ############
                Bls[0] = dXn*dPhi
                Bls[1] = dYn*dPhi
            
            Eg.addA(Als)
            Eg.addB(Bls)

    def __private__buildingInternal(self):

        for face_i in range(self._numberOfBoundaryFaces, self._numberOfFaces):
        
                #Préparation
                elem = self._mesh.get_face_to_elements(face_i)
                Eg = self._elements[elem[0]]
                Ed = self._elements[elem[1]]

                dX = Ed._center[0] - Eg._center[0]
                dY = Ed._center[1] - Eg._center[1]

                #Matrices temporaires
                Als = np.zeros((2,2))
                Bls = np.zeros(2)

                #Matrice ATA
                Als[0,0] = dX**2
                Als[1,0] = dX*dY
                Als[0,1] = dY*dX
                Als[1,1] = dY**2


                #Matrice B
                Bls[0] = dX*(Ed._value - Eg._value)
                Bls[1] = dY*(Ed._value - Eg._value)

                #Element droit
                Ed.addA(Als)
                Ed.addB(Bls)

                #Element gauche
                Eg.addA(Als) 
                Eg.addB(Bls)
            




