

import numpy as np


from  element import Element


class CL:
    def __init__(self, parameters : dict[int,(str,any)]):

        
        """
        Constructeur de la classe CL.

        Parameters
        ----------
        parameters : dict[int,(str,float)]
            Dictionnaire qui contient les paramètres de la condition au bord pour chaque face.
            Les clés sont les tags de chaque bord et les valeurs sont des tuples (str, float) contenant
            le type de condition ('D' pour Dirichlet, 'N' pour Neumann, 'L' pour Libre) et la valeur
            de la condition (par exemple, la valeur du champ pour une condition de Dirichlet).
            Ex : param[1]=('D',2)
        Returns
        -------
        None
        """

        self.CLConfig = parameters
    
    def calculCL_ATA(self,tag : int,face : int,Eg : Element)->np.ndarray:
        
        """
        Calcul de la contribution de la condition au bord pour une face dans la matrice ATA.

        Parameters
        ----------
        tag : int
            Tag de la face externe.
        face : int
            Numéro de la face.
        Eg : Element
            Élément concerné.

        Returns
        -------
        Ald : ndarray
            Contribution de la condition au bord dans la matrice ATA.
        """
        
        if self.CLConfig[tag][0] == "D": #Dirichlet

        
            return self.__private_Dirichlet_ATA(face,Eg)
        
        elif self.CLConfig[tag][0] == "N": #Neumann

        
            return self.__private_Neumann_ATA(face,Eg)
        
        else: #Libre
            
            return np.zeros((2,2))
        
    def calculCL_B(self,tag : int,face : int,Eg : Element)->np.ndarray:
        
        """
        Calcul de la contribution de la condition au bord pour une face dans le second membre.

        Parameters
        ----------
        tag : int
            Tag de la face externe.
        face : int
            Numéro de la face.
        Eg : Element
            Élément concerné.

        Returns
        -------
        Bld : ndarray
            Contribution de la condition au bord dans le second membre.
        """
        if self.CLConfig[tag][0] == "D": #Dirichlet
            
            return self.__private_Dirichlet_B(tag,face,Eg)

        elif self.CLConfig[tag][0] == "N": #Neumann
            
            return self.__private_Neumann_B(tag,face,Eg)

        else: #Libre
        
            return np.zeros(2)

    
    def __private_Dirichlet_ATA(self,face:int,Eg:Element)->np.ndarray:

        """
        Calcul de la contribution de la condition de Dirichlet pour une face dans la matrice ATA.

        Parameters
        ----------
        face : int
            Numéro de la face.
        Eg : Element
            Élément concerné.

        Returns
        -------
        Al : ndarray
            Contribution de la condition de Dirichlet dans la matrice ATA.
        """
        FaceCenter = Eg.calculFaceCenter(face)
        EgCenter = Eg.get_Coord()

        DX = FaceCenter[0] - EgCenter[0]
        DY = FaceCenter[1] - EgCenter[1]

        Al = np.zeros((2,2))
        Al[0,0] = DX**2
        Al[1,0] = DX*DY
        Al[0,1] = DY*DX
        Al[1,1] = DY**2

        return Al
    
    def __private_Neumann_ATA(self,face:int,Eg:Element)->np.ndarray:

        """
        Calcul de la contribution de la condition de Neumann pour une face dans la matrice ATA.

        Parameters
        ----------
        face : int
            Numéro de la face.
        Eg : Element
            Élément concerné.

        Returns
        -------
        Al : ndarray
            Contribution de la condition de Neumann dans la matrice ATA.
        """
        FaceCenter = Eg.calculFaceCenter(face)
        EgCenter = Eg.get_Coord()
        Normal = Eg.calculFaceNormal(face)


        DX = FaceCenter[0] - EgCenter[0]
        DY = FaceCenter[1] - EgCenter[1]

        DXn = (DX*Normal[0]+DY*Normal[1])*Normal[0]
        DYn = (DX*Normal[0]+DY*Normal[1])*Normal[1]

        Al = np.zeros((2,2))
        Al[0,0] = DXn**2
        Al[1,0] = DXn*DYn
        Al[0,1] = DYn*DXn
        Al[1,1] = DYn**2

        return Al
    
    def __private_Dirichlet_B(self,tag:int,face:int,Eg:Element)->np.ndarray:

        """
        Calcul de la contribution de la condition de Dirichlet pour une face dans le second membre.

        Parameters
        ----------
        tag : int
            Tag de la face externe.
        face : int
            Numéro de la face.
        Eg : Element
            Élément concerné.

        Returns
        -------
        Bld : ndarray
            Contribution de la condition de Dirichlet dans le second membre.
        """
        
        FaceCenter = Eg.calculFaceCenter(face)
        EgCenter = Eg.get_Coord()

        DX = FaceCenter[0] - EgCenter[0]
        DY = FaceCenter[1] - EgCenter[1]

        B = np.zeros(2)
        PhiA = 0

        
        if callable(self.CLConfig[tag][1]):
            PhiA = self.CLConfig[tag][1](FaceCenter[0],FaceCenter[1])
        else:
            PhiA = self.CLConfig[tag][1]
        
        B[0] = DX*(PhiA-Eg.get_value())
        B[1] = DY*(PhiA-Eg.get_value())

        return B
    
    def __private_Neumann_B(self,tag:int,face:int,Eg:Element)->np.ndarray:

        """
        Calcul de la contribution de la condition de Neumann pour une face dans le second membre.

        Parameters
        ----------
        tag : int
            Tag de la face externe.
        face : int
            Numéro de la face.
        Eg : Element
            Élément concerné.

        Returns
        -------
        Bld : ndarray
            Contribution de la condition de Neumann dans le second membre.
        """
        FaceCenter = Eg.calculFaceCenter(face)
        EgCenter = Eg.get_Coord()
        Normal = Eg.calculFaceNormal(face)

        DX = FaceCenter[0] - EgCenter[0]
        DY = FaceCenter[1] - EgCenter[1]

        DXn = (DX*Normal[0]+DY*Normal[1])*Normal[0]
        DYn = (DX*Normal[0]+DY*Normal[1])*Normal[1]

        B = np.zeros(2)
        
        #Gradient de la fonction de contrôle exacte ou valeur constante
            
        if callable(self.CLConfig[tag][1]):
            G = self.CLConfig[tag][1](FaceCenter[0],FaceCenter[1])
            GNormal = G@Normal
        else:
            GNormal = self.CLConfig[tag][1]


        B[0] = DXn*(DX*Normal[0]+DY*Normal[1])*GNormal
        B[1] = DYn*(DX*Normal[0]+DY*Normal[1])*GNormal

        return B


        


