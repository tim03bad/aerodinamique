
from calendar import c
import numpy as np
import copy as cpy



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
        self._pressureGrad = self.computePressureGradient()


        #Champs de vitesse (initialisé à 0)
        self._velocityField = np.zeros((len(elements), 2))

        #Stockaga matrice Au pour calcul de rhie-crow
        self._Au = np.zeros((len(self._elements),len(self._elements)))


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
        converged = False
        iteration = 0

        while not converged:
            if self._verbose:
                print('Iteration ' + str(iteration))

            solverX = Solver(self._mesh, self._elements,self._velocityField,self._CLx,self._rho,self._mu,-self._pressureGrad[:,0],self._pressureGrad,scheme='central',verbose=self._verbose)
            solverY = Solver(self._mesh, self._elements,self._velocityField,self._CLy,self._rho,self._mu,-self._pressureGrad[:,1],self._pressureGrad,scheme='central',verbose=self._verbose)

            #résolution des champs de vitesse
            u = solverX.solve()
            v = solverY.solve()

            #relaxation
            alpha = 0.8
            uf = alpha*u + (1-alpha)*self._velocityField[:,0]
            vf = alpha*v + (1-alpha)*self._velocityField[:,1]

            #Calcul des résidus
            Ru, Rv = self.computeResidual(solverX,solverY)
            if self._verbose:
                print("Ru = {}, Rv = {}".format(Ru,Rv))
            
            if Ru <= Rmax and Rv <= Rmax:
                converged = True
                #Stockage de Au pour le calcul éventuelle de rhie-crow
                self._Au = solverX._A

            #Mise à jour des champs de vitesse
            self._velocityField[:,0] = uf
            self._velocityField[:,1] = vf

            iteration += 1

        return self._velocityField
    
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
        #On réalise une copie des elements
        elems = cpy.deepcopy(self._elements)
        pf = np.zeros(len(elems))
        for e in elems:
            e._value = self._pressureField(e._center[0],e._center[1])
            pf[e._index] = e._value

        CL = {
            0:('D',self._pressureField),
            1:('D',self._pressureField),
            2:('D',self._pressureField),
            3:('D',self._pressureField)
        }

        gradSolver = LeastSquareSolver(self._mesh,elems,CL,verbose=self._verbose)
        gradSolver.solve()

        gradP = np.zeros((len(self._elements), 2))
        for e in elems:
            gradP[e._index] = cpy.deepcopy(e._grad)


        del elems

        if self._verbose:
            print("\n ---------  Successfully computed pressure gradient --------- \n")
            util().plotField(self._mesh,pf,"Champs de pression")
        return gradP
    
    
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

        Gg = self._pressureGrad[Eg._index]
        Gd = self._pressureGrad[Ed._index]

        Ug = self._velocityField[Eg._index]
        Ud = self._velocityField[Ed._index]

        Pg = self._pressureField(Eg._center[0],Eg._center[1])
        Pd = self._pressureField(Ed._center[0],Ed._center[1])

        c1 =  0.5*(Ug+Ud)@N
        c2 = 0.5*(Ag/ag+Ad/ad)*(Pg - Pd)/dKsi
        c3 = 0.5*((Ag/ag)*Gg + (Ad/ad)*Gd)@e_ksi

        return c1 + c2 + c3
    
    def testRhieCrow(self):
        """
        Test the Rhie-Crow interpolation on internal faces.

        Notes
        -----
        Prints the result of the Rhie-Crow interpolation on internal faces, and
        compares it with a simple average of the velocity values at the center of
        the neighboring elements.
        """
        print("-- Test Rhie-Crow on internal faces --")
        for face_i in range(self._mesh.number_of_boundary_faces, self._mesh.number_of_faces):

            elems = self._mesh.get_face_to_elements(face_i)

            Eg = self._elements[elems[0]]
            Ed = self._elements[elems[1]]

            A = self._Au
            N = util.getFaceNormal(self._mesh, face_i)
            Vmoysimple = 0.5*(self._velocityField[Eg._index] + self._velocityField[Ed._index])@N
            rc = self.rhieCrow(face_i,Eg,Ed,A)

            print(" ->Face {} : RC {:.4f} | Vmoysimple {:.4f} ==> {:.4f}".format(face_i,rc,Vmoysimple,(rc-Vmoysimple)))


        
        



