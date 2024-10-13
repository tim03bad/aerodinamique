
import numpy as np
from sympy import ring


from mesh import Mesh

from element import Element
from solver import Solver
from utilities import Utilities as util

class Momentum2DSolver():

    def __init__(self, mesh: Mesh, elements: list[Element], CL: dict,rho:float, mu: float,pressureFieldGrad,scheme: str = 'upwind',verbose: bool = False):

        self._verbose = verbose

        #Reference au maillage
        self._mesh = mesh

        #Réference aux éléments
        self._elements = elements

        #Conditions aux limites
        self._CL = CL

        #Parametres thermodynamiques
        self._rho = rho
        self._mu = mu #Viscosité dynamique

        #Terme source (champs de pression)
        self._pressureFieldGrad = pressureFieldGrad
        if callable(self._pressureFieldGrad[0]):
            self._pressureFieldGrad_x = lambda x,y: -self._pressureFieldGrad[0](x,y)
        else:
            self._pressureFieldGrad_x = lambda x,y: -self._pressureFieldGrad[0]

        if callable(self._pressureFieldGrad[1]):
            self._pressureFieldGrad_y = lambda x,y: -self._pressureFieldGrad[1](x,y)
        else:
            self._pressureFieldGrad_y = lambda x,y: -self._pressureFieldGrad[1]
    

        #Schemes
        self._scheme = 'upwind'; #Default central scheme
        if scheme == 'central':
            self._scheme = 'central' #Switch to central scheme

        #Champs de vitesse (initialisé à 0)
        self._velocityField = np.zeros((len(elements), 2))


    def solve(self,Rmax:float):
        converged = False
        iteration = 0

        while not converged:
            
            print('Iteration ' + str(iteration))

            solverX = Solver(self._mesh, self._elements,self._velocityField,self._CL,self._rho,self._mu,self._pressureFieldGrad_x,self._scheme,verbose=True)
            solverY = Solver(self._mesh, self._elements,self._velocityField,self._CL,self._rho,self._mu,self._pressureFieldGrad_y,self._scheme,verbose=True)

            #résolution des champs de vitesse
            u = solverX.solve()
            v = solverY.solve()

            #relaxation
            alpha = 0.8
            uf = alpha*u + (1-alpha)*self._velocityField[:,0]
            vf = alpha*v + (1-alpha)*self._velocityField[:,1]

            #Calcul des résidus
            Ru, Rv = self.computeResidual(solverX,solverY)
            print("Ru = {}, Rv = {}".format(Ru,Rv))
            
            if Ru <= Rmax and Rv <= Rmax:
                converged = True

            #Mise à jour des champs de vitesse
            self._velocityField[:,0] = uf
            self._velocityField[:,1] = vf

            iteration += 1

        return self._velocityField
    
    def computeResidual(self,solverX: Solver,solverY: Solver):

        Ru = solverX._A@self._velocityField[:,0] - solverX._B
        Rv = solverY._A@self._velocityField[:,1] - solverY._B

        return np.linalg.norm(Ru), np.linalg.norm(Rv)



