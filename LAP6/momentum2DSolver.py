

import numpy as np


from mesh import Mesh

from element import Element
from solver import Solver
from leastSquareSolver import LeastSquareSolver
from utilities import Utilities as util

class Momentum2DSolver():

    def __init__(self, mesh: Mesh, elements: list[Element], CLx: dict, CLy: dict,rho:float, mu: float,pressureField,verbose: bool = False):

        """
        Initialize the Momentum2DSolver with the given parameters.

        Parameters:
        ----------
        mesh (Mesh): The mesh object representing the computational domain.
        elements (list[Element]): List of element objects in the mesh.
        CLx (dict): Boundary conditions for the x-direction.
        CLy (dict): Boundary conditions for the y-direction.
        rho (float): Density of the fluid.
        mu (float): Dynamic viscosity of the fluid.
        pressureField: pressure field, can be a callable (2D function) or a constant value.
        verbose (bool): If True, enables verbose output for debugging.

        Initializes velocity field to zero and computes pressure field gradients.
        """
        self._verbose = verbose

        #Reference au maillage
        self._mesh = mesh

        #Réference aux éléments
        self._elements = elements

        #Conditions aux limites
        self._CLx = CLx
        self._CLy = CLy
        #Parametres thermodynamiques
        self._rho = rho
        self._mu = mu #Viscosité dynamique

        #Terme source (champs de pression)
        self._pressureField = pressureField
        self.computePressureGradient()

        self._Pprime = np.zeros(self._mesh.get_number_of_elements())


        #Champs de vitesse (initialisé à 0)
        self._velocityField = np.zeros((len(elements), 2))
        self._Vrc = np.zeros(self._mesh.get_number_of_faces())

        #Stockaga matrice Au pour calcul de rhie-crow
        self._Au = np.zeros((len(self._elements),len(self._elements)))

        #Donnée Face (centre et normal)
        self._FaceCenters = np.zeros((self._mesh.get_number_of_faces(),2))
        self._FaceNormals = np.zeros((self._mesh.get_number_of_faces(),2))
        for face_i in range(self._mesh.get_number_of_faces()):
            self._FaceCenters[face_i] = util.getFaceCenter(self._mesh, face_i)
            self._FaceNormals[face_i] = util.getFaceNormal(self._mesh, face_i)


    def solve(self,Rmax:float):
        """
        Solves the momentum equation for the given mesh, elements, boundary conditions and physical parameters.

        Parameters
        ----------
        Rmax (float): Maximum residual for convergence.

        Returns
        -------
        velocityField (numpy array): The computed velocity field.

        Notes
        -----
        Solves the momentum equation using a segregated approach with a SIMPLE-like relaxation scheme. The velocity field is relaxed using a linear relaxation scheme with a relaxation factor of 0.8. Convergence is checked using the maximum residual of the x and y components of the momentum equation.
        """
        

        solverX = Solver(self._mesh, self._elements,0,self._CLx,self._rho,self._mu,scheme='central',verbose=self._verbose)
        solverY = Solver(self._mesh, self._elements,1,self._CLy,self._rho,self._mu,scheme='central',verbose=self._verbose)

        #résolution des champs de vitesse
        u = solverX.solve()
        v = solverY.solve()


        self._Au = solverX._A

        #Calcul de la vitesse débitante non corrigée avec Rhie-Crow
        #External face
        for face_i in range(self._mesh.number_of_boundary_faces):
            elems = self._mesh.get_face_to_elements(face_i)
            Eg = self._elements[elems[0]]

            self._Vrc[face_i] = self.RhieCrowExternal(face_i,Eg,self._Au)

        #internal faces
        for face_i in range(self._mesh.number_of_boundary_faces, self._mesh.number_of_faces):
            elems = self._mesh.get_face_to_elements(face_i)
            Eg = self._elements[elems[0]]    
            Ed = self._elements[elems[1]]

            self._Vrc[face_i] = self.rhieCrow(face_i,Eg,Ed,self._Au)

        util.plotFaceVelocity(self._mesh,self._Vrc,self._FaceCenters,self._FaceNormals,'Vitesse débitante non corrigée')

        #Calcul de la correction de pression
        self._Pprime,dfi = self.pressureCorrection(self._Vrc)


        #Calcul de la vitesse débitante corrigée par la correction de pression
        for face_i in range(self._mesh.number_of_boundary_faces):
            elems = self._mesh.get_face_to_elements(face_i)

            self._Vrc[face_i] = self._Vrc[face_i] + dfi[face_i]*self._Pprime[elems[0]]


        for face_i in range(self._mesh.number_of_boundary_faces, self._mesh.number_of_faces):
            elems = self._mesh.get_face_to_elements(face_i)

            self._Vrc[face_i] = self._Vrc[face_i] + dfi[face_i]*(self._Pprime[elems[0]]-self._Pprime[elems[1]])     


        velocityField = np.column_stack((u,v))
        util.plotField(self._mesh,self._Pprime,'Correction de pression')
        util.plotFaceVelocity(self._mesh,self._Vrc,self._FaceCenters,self._FaceNormals,'Vitesse débitante corrigée')


        #Calcul de la divergence du champ de vitesse débitante corrigée
        D = self.getDivergence(self._Vrc)
        print("Divergence : {}".format(D))
        

        return velocityField
    
    def computeResidual(self,solverX: Solver,solverY: Solver):

        """
        Compute the residual of the momentum equation in the x and y directions.

        Parameters
        ----------
        solverX : Solver
            The solver for the x-component of the momentum equation
        solverY : Solver
            The solver for the y-component of the momentum equation

        Returns
        -------
        tuple
            The residuals of the x and y components of the momentum equation
        """
        Ru = solverX._A@self._velocityField[:,0] - solverX._B
        Rv = solverY._A@self._velocityField[:,1] - solverY._B

        return np.linalg.norm(Ru), np.linalg.norm(Rv)
    
    def computePressureGradient(self):
        """
        Compute the pressure gradient from a given pressure field.

        Parameters
        ----------
        None

        Returns
        -------
        gradP (numpy array): The computed pressure gradient.

        Notes
        -----
        The pressure gradient is computed using a least square approach. The pressure field is first evaluated at the center of each element, then the gradient is computed using a least square solver. The result is a 2D array where each row corresponds to an element and each column corresponds to the x and y components of the pressure gradient.
        """
        if self._verbose:
            print("\n ---------  Computing pressure gradient --------- \n")

        for e in self._elements:
            e.setPressure(self._pressureField)

        CL = {
            0:('D',self._pressureField),
            1:('D',self._pressureField),
            2:('D',self._pressureField),
            3:('D',self._pressureField)
        }

        gradSolver = LeastSquareSolver(self._mesh,self._elements,CL,field="P",verbose=self._verbose)
        gradSolver.solve()
        if self._verbose:
            print("\n ---------  Successfully computed pressure gradient --------- \n")

    
    
    def rhieCrow(self,face_i:int,Eg:Element,Ed:Element,A:np.ndarray):

        """
        Compute the Rhie-Crow interpolation for the face face_i.

        Parameters
        ----------
        face_i : int
            Index of the face.
        Eg : Element
            Element on the left of the face.
        Ed : Element
            Element on the right of the face.
        A : numpy array
            Diagonal of the matrix A.

        Returns
        -------
        float
            The interpolated value of the velocity field at the center of the face.

        Notes
        -----
        The Rhie-Crow interpolation is a method to compute the velocity at the center of a face from the velocity values at the center of the neighboring elements. The interpolated value is computed as a weighted average of the velocity values at the center of the elements, where the weights are the areas of the elements divided by the distance between the center of the elements and the center of the face. The result is a scalar value that represents the velocity at the center of the face.
        """
        N = util.getFaceNormal(self._mesh, face_i)
        dKsi = np.linalg.norm(Ed._center - Eg._center)
        e_ksi = (Ed._center - Eg._center)/dKsi

        DAU = np.diag(A)
        ag = DAU[Eg._index]
        ad = DAU[Ed._index]

        Ag = Eg._area
        Ad = Ed._area

        Gg = Eg._gradP
        Gd = Ed._gradP

        Ug = Eg._V
        Ud = Ed._V

        Pg = Eg._P
        Pd = Ed._P


        c1 =  0.5*(Ug+Ud)@N
        c2 = 0.5*(Ag/ag+Ad/ad)*(Pg - Pd)/dKsi
        c3 = 0.5*((Ag/ag)*Gg + (Ad/ad)*Gd)@e_ksi

        return c1 + c2 + c3
    
    def RhieCrowExternal(self,face_i:int,Eg:Element,A:np.ndarray):

        
        """
        Compute the Rhie-Crow interpolation for the external face face_i.

        Parameters
        ----------
        face_i : int
            Index of the face.
        Eg : Element
            Element on the left of the face.
        A : numpy array
            Diagonal of the matrix A.

        Returns
        -------
        float
            The interpolated value of the velocity field at the center of the face.

        Notes
        -----
        This function is used to compute the Rhie-Crow interpolation for the external faces of the mesh. The interpolated value is computed as a weighted average of the velocity values at the center of the element and the boundary value, where the weights are the areas of the element divided by the distance between the center of the element and the center of the face. The result is a scalar value that represents the velocity at the center of the face.

        """
        N = util.getFaceNormal(self._mesh, face_i)
        tag = self._mesh.get_boundary_face_to_tag(face_i)

        if self._CLx[tag][0] == 'D':
            FC = util.getFaceCenter(self._mesh, face_i)

            if callable(self._CLx[tag][1]):
                u = self._CLx[tag][1](FC[0],FC[1])
            else:
                u = self._CLx[tag][1]

            if callable(self._CLy[tag][1]):
                v = self._CLy[tag][1](FC[0],FC[1])
            else:
                v = self._CLy[tag][1]


            return u*N[0] + v*N[1]
        
        elif self._CLx[tag][0] == 'N':
            #Pression imposée
            FaceCenter = util.getFaceCenter(self._mesh, face_i)
            dKsi = np.linalg.norm(FaceCenter - Eg._center)
            e_ksi = (FaceCenter - Eg._center)/dKsi

            up = Eg._V
            deltaVp = Eg._area
            ap = A[Eg._index,Eg._index]

            Ps = self._pressureField(FaceCenter[0],FaceCenter[1])
            Pp = Eg._P

            gradP = Eg._gradP

            return up@N + (deltaVp/ap)*(Pp - Ps)/dKsi + (deltaVp/ap)*gradP@e_ksi



    def dfi(self,Eg:Element,Ed:Element,dKsi:float):
        """
        Compute the interpolation factor dfi for a given face between two elements.

        Parameters
        ----------
        Eg : Element
            Element on the left side of the face.
        Ed : Element
            Element on the right side of the face.
        dKsi : float
            Distance between the centers of the two elements.

        Returns
        -------
        float
            The interpolation factor dfi, representing the weighted average based on the areas and coefficients
            of the elements, divided by the distance dKsi.

        Notes
        -----
        The interpolation factor is used in the Rhie-Crow interpolation to compute velocity correction terms.
        It is derived from the areas of the elements and the diagonal coefficients of the matrix A.
        """
        SP = Eg._area
        SA = Ed._area
        ap = self._Au[Eg._index,Eg._index]
        aa = self._Au[Ed._index,Ed._index]

        return 0.5*(SP/ap + SA/aa)/dKsi
    
    
    def pressureCorrection(self,Vrc:np.ndarray):
        """
        Computes the pressure correction for the velocity field using the Rhie-Chow interpolation method.

        Parameters
        ----------
        Vrc : np.ndarray
            Array representing the velocity at the faces of the mesh.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - The pressure correction field computed from the matrix equation Mp * P' = B.
            - The interpolation factor dfi for each face, used in the Rhie-Chow interpolation.

        Notes
        -----
        The method constructs the matrices Mp and B based on the internal and external faces of the mesh.
        For boundary faces, the method accounts for solid walls and entrances by modifying the matrix B.
        For external faces, the method constructs the matrix Mp using the interpolation factors computed
        from the areas and diagonal coefficients of the elements adjacent to each face.
        """
        Mp = np.zeros((self._mesh.get_number_of_elements(),self._mesh.get_number_of_elements()))
        B = np.zeros(self._mesh.get_number_of_elements())
        dfi = np.zeros(self._mesh.get_number_of_faces())

        # Contruction des matrices Mp et B
        #internal faces
        for face_i in range(self._mesh.number_of_boundary_faces):
            tag = self._mesh.get_boundary_face_to_tag(face_i)
            elems = self._mesh.get_face_to_elements(face_i)

            Eg = self._elements[elems[0]]

            dAi = util.getFaceLength(self._mesh, face_i)
            FaceCenter = util.getFaceCenter(self._mesh, face_i)
            dKsi = np.linalg.norm(Eg._center - FaceCenter)

            #Parois solide ou entrée
            if self._CLx[tag][2] == 'E' or self._CLx[tag][2] == 'W':
                #print("Face de paroi | entrée ")

                B[elems[0]] -= self._rho*Vrc[face_i]*dAi
            
            #Sortie
            elif self._CLx[tag][2] == 'S':

                Mp[elems[0],elems[0]] += self._rho*dAi*Eg._area/(self._Au[Eg._index,Eg._index]*dKsi)
                B[elems[0]] -= self._rho*Vrc[face_i]*dAi

                dfi[face_i] = Eg._area/(self._Au[Eg._index,Eg._index]*dKsi)


        #external faces
        for face_i in range(self._mesh.number_of_boundary_faces, self._mesh.number_of_faces):
            elems = self._mesh.get_face_to_elements(face_i)
            Eg = self._elements[elems[0]]
            Ed = self._elements[elems[1]]
            dKsi = np.linalg.norm(Eg._center - Ed._center)
            dAi = util.getFaceLength(self._mesh, face_i)

            M = self._rho*self.dfi(Eg,Ed,dKsi)*dAi

            Mp[elems[0],elems[0]] += M
            Mp[elems[0],elems[1]] -= M
            Mp[elems[1],elems[0]] -= M
            Mp[elems[1],elems[1]] += M

            B[elems[0]] -= self._rho*Vrc[face_i]*dAi
            B[elems[1]] += self._rho*Vrc[face_i]*dAi

            dfi[face_i] = self.dfi(Eg,Ed,dKsi)

        #Résolution et retour de la fonction
        return np.linalg.solve(Mp,B),dfi
    

    def getDivergence(self,Vrc:np.ndarray):

        D = np.zeros(self._mesh.get_number_of_faces())

        for face_i in range(self._mesh.number_of_faces):
            elems = self._mesh.get_face_to_elements(face_i)

            dAi = util.getFaceLength(self._mesh, face_i)

            D[elems[0]] -=  self._rho*Vrc[face_i]*dAi
            D[elems[1]] +=  self._rho*Vrc[face_i]*dAi

        return D



        
        



