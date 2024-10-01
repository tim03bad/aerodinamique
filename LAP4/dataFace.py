class dataFace:
    def __init__(self,Xa,Xb,Ya,Yb,XA,XP,YA,YP):
        """
        Constructeur de l'objet dataFace

        Parameters
        ----------
        Xa,Ya : float
            Coordonn es du premier point de la face
        Xb,Yb : float
            Coordonn es du deuxi me point de la face
        XA,YA : float
            Coordonn es du centre de l' l ment
        XP,YP : float
            Coordonn es du centre de l' l ment adjacent

        Sert de conteneur pour les coordonnées des points utile au calcul
        """
        self.Xa = Xa
        self.Xb = Xb
        self.Ya = Ya
        self.Yb = Yb
        self.XA = XA
        self.XP = XP
        self.YA = YA
        self.YP = YP

    def __str__(self):
        """
        Affichage des donn es de l'objet
        """
        return "Xa : {:.3f}, Xb : {:.3f}, Ya : {:.3f}, Yb : {:.3f}\nXA : {:.3f}, XP : {:.3f}, YA : {:.3f}, YP : {:.3f}".format(self.Xa,self.Xb,self.Ya,self.Yb,self.XA,self.XP,self.YA,self.YP)
