import numpy as np
from sqlalchemy import func

from element import Element

class Champ:

    def __init__(self, champ : function, grad : function):
        """
        Constructeur de l'objet Champ

        Parameters
        ----------
        champ : function
            Fonction qui prend en argument deux valeurs (x,y) et qui renvoie la valeur du champ au point (x,y)
        grad : function
            Fonction qui prend en argument deux valeurs (x,y) et qui renvoie la valeur du gradient du champ au point (x,y)

        Returns
        -------
        None
        """
        self.champ = champ
        self.grad  = grad


    def set_ElementsValue(self,elements : list[Element]):
        """
        Fixe la valeur de chaque element de la liste des elements avec la valeur du champ a cette position

        Parameters
        ----------
        elements : list[Element]
            Liste des elements dont on veut fixer la valeur

        Returns
        -------
        None
        """
        for element in elements:
            coord = element.get_Coord();
            element.set_value(self.get_value(coord[0],coord[1]))


    def set_function(self, champ : function):
        self.function = champ

    def set_grad(self, grad : function):
        self.grad = grad
        

    def get_value(self, x : float, y : float):
        return self.function(x,y)
    
    def get_grad(self, x : float, y : float):
        return self.grad(x,y)