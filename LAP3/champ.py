import numpy as np
from sqlalchemy import func

from element import Element

class Champ:

    def __init__(self, champ : function, grad : function):
        self.champ = champ
        self.grad  = grad


    def set_ElementValue(self,elements : list[Element]):
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