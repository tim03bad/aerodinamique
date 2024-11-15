#Python libraries
import numpy as np
import pyvista as pv
import pyvistaqt as pvQt
import matplotlib.pyplot as plt
import sympy as sp

#Mesh libraries
from mesh import Mesh
from meshPlotter import MeshPlotter
from meshConnectivity import MeshConnectivity
from meshGenerator import MeshGenerator

#Custom classes
from element import Element



class Utilities:

    @staticmethod
    def getFaceCenter(mesh: Mesh, face_i):
        """
        Calcule le centre d'une face du maillage.
        
        Parameters
        ----------
        mesh: Mesh
            Objet maillage.
        face_i: int
            Index de la face dont on souhaite calculer le centre.
        
        Returns
        -------
        ndarray, shape (2,)
            Coordonnées du centre de la face.
        """
        
        nodes = mesh.get_face_to_nodes(face_i)
        NodesCoords = np.zeros((len(nodes), 2))
        for i in range(len(nodes)):
            NodesCoords[i, 0] = mesh.get_node_to_xcoord(nodes[i])
            NodesCoords[i, 1] = mesh.get_node_to_ycoord(nodes[i])
        return np.mean(NodesCoords, axis=0)
    


    @staticmethod
    def getFaceNormal(mesh: Mesh, face_i):
        """
        Calcule le vecteur normal d'une face du maillage.
        
        Parameters
        ----------
        mesh: Mesh
            Objet maillage.
        face_i: int
            Index de la face dont on souhaite calculer le vecteur normal.
        
        Returns
        -------
        ndarray, shape (2,)
            Vecteur normal de la face.
        """
        nodes = mesh.get_face_to_nodes(face_i)
        NodesCoords = np.zeros((len(nodes), 2))
        for i in range(len(nodes)):
            NodesCoords[i, 0] = mesh.get_node_to_xcoord(nodes[i])
            NodesCoords[i, 1] = mesh.get_node_to_ycoord(nodes[i])
        
        Temp = NodesCoords[1] - NodesCoords[0]
        Temp = np.array([Temp[1], -Temp[0]])
        return Temp/np.linalg.norm(Temp)
    

    @staticmethod
    def getFaceLength(mesh: Mesh, face_i):
        """
        Calcule la longueur d'une face du maillage.
        
        Parameters
        ----------
        mesh: Mesh
            Objet maillage.
        face_i: int
            Index de la face dont on souhaite calculer la longueur.
        
        Returns
        -------
        double
            Longueur de la face.
        """
        nodes = mesh.get_face_to_nodes(face_i)
        NodesCoords = np.zeros((len(nodes), 2))
        for i in range(len(nodes)):
            NodesCoords[i, 0] = mesh.get_node_to_xcoord(nodes[i])
            NodesCoords[i, 1] = mesh.get_node_to_ycoord(nodes[i])

        return np.linalg.norm(NodesCoords[1] - NodesCoords[0])
    

    @staticmethod
    def getFaceUnitVector(mesh: Mesh, face_i):
        """
        Calcule le vecteur unitaire d'une face du maillage.
        
        Parameters
        ----------
        mesh: Mesh
            Objet maillage.
        face_i: int
            Index de la face dont on souhaite calculer le vecteur unitaire.
        
        Returns
        -------
        ndarray, shape (2,)
            Vecteur unitaire de la face.
        """
        nodes = mesh.get_face_to_nodes(face_i)
        NodesCoords = np.zeros((len(nodes), 2))
        for i in range(len(nodes)):
            NodesCoords[i, 0] = mesh.get_node_to_xcoord(nodes[i])
            NodesCoords[i, 1] = mesh.get_node_to_ycoord(nodes[i])

        Temp = NodesCoords[1] - NodesCoords[0]

        return Temp/np.linalg.norm(Temp)
    


    @staticmethod
    def prepareMesh(mesh_param : dict,bounds : list[int],value):

        """
        Prépare un maillage pour la résolution de l'équation de Poisson.
        
        Parameters
        ----------
        mesh_param : dict
            Dictionnaire contenant les paramètres du maillage à générer. Les clés sont :
            * 'mesh_type': string, optionnel
                Type des éléments du maillage. Les valeurs possibles sont :
                - 'TRI' : triangles (par défaut)
                - 'QUAD' : quadrilatères
                - 'MIX' : mixe de triangles et de quadrilatères
            * 'lc': double
                taille caractéristique des éléments du maillage (maillages non-structurés).
            * 'Nx': integer ; 'Nx': integer
                Nombre de cellules selon les axes x et y (maillages transfinis).
            
        value: function or double
            Fonction ou valeur associée aux éléments du maillage. Si fonction, elle prendra
            comme arguments les coordonnées (x, y) du centre de l'élément et renverra
            la valeur associée à cet élément.
        
        Returns
        -------
        mesh: Mesh
            Objet Mesh contenant le maillage généré.
        Elements: list[Element]
            Liste des éléments du maillage.
        """
        mesher = MeshGenerator(verbose=False)
        mesh = mesher.rectangle(bounds, mesh_param)
        conec = MeshConnectivity(mesh,verbose=False)
        conec.compute_connectivity()

        #Generation des éléments associés
        Elements = Utilities.generateElements(mesh)

        return mesh, Elements
    



    @staticmethod
    def generateElements(mesh: Mesh):
        """
        Génère les éléments associés au maillage.
        
        Parameters
        ----------
        mesh: Mesh
            Objet Mesh contenant le maillage.
        value: function or double
            Fonction ou valeur associée aux éléments du maillage. Si fonction, elle prendra
            comme arguments les coordonnées (x, y) du centre de l'élément et renverra
            la valeur associée à cet élément.
        
        Returns
        -------
        Elements: list[Element]
            Liste des éléments du maillage.
        """
        Elements = []
        for i in range(mesh.get_number_of_elements()):
            Elements.append(Element(mesh, i))
        return Elements
    


    @staticmethod
    def getGradVectors(mesh: Mesh, Elements: list[Element],gradPhi):

        """
        Renvoie les gradients numérique et analytique associés aux éléments du maillage.
        
        Parameters
        ----------
        mesh: Mesh
            Objet Mesh contenant le maillage.
        Elements: list[Element]
            Liste des éléments du maillage.
        gradPhi: function
            Fonction renvoyant le gradient analytique associé à un élément.
        
        Returns
        -------
        GradN: ndarray, shape (mesh.get_number_of_elements(),2)
            Gradient numérique associé à chaque élément.
        GradA: ndarray, shape (mesh.get_number_of_elements(),2)
            Gradient analytique associé à chaque élément.
        """
        GradN = np.zeros((mesh.get_number_of_elements(),2)) #Gradient numérique
        GradA = np.zeros((mesh.get_number_of_elements(),2)) #Gradient analytique

        for e in Elements:
            GradA[e._index] = gradPhi(e._center[0], e._center[1])
            GradN[e._index] = e._grad

        return GradN, GradA
    



    @staticmethod
    def getGradNorm(Mesh: Mesh, Elements: list[Element], gradPhi):

        """
        Renvoie les normes des gradients numérique et analytique associés aux éléments du maillage.
        
        Parameters
        ----------
        Mesh: Mesh
            Objet Mesh contenant le maillage.
        Elements: list[Element]
            Liste des éléments du maillage.
        gradPhi: function
            Fonction renvoyant le gradient analytique associé à un élément.
        
        Returns
        -------
        GradNorm: tuple[ndarray, ndarray]
            Tuple contenant les normes des gradients numérique et analytique associés
            à chaque élément.
        """
        GradN, GradA = Utilities.getGradVectors(Mesh, Elements, gradPhi)
        return np.linalg.norm(GradN, axis=1), np.linalg.norm(GradA, axis=1)
    



    @staticmethod
    def plotField(mesh: Mesh,field,title:str,fieldA = 0):
        """
        Affiche le champ scalaire associé au maillage.
        
        Parameters
        ----------
        mesh: Mesh
            Objet Mesh contenant le maillage.
        field: ndarray, shape (mesh.get_number_of_elements(),)
            Champ scalaire associé à chaque élément.
        title: str
            Titre associé au champ scalaire.
        fieldA: ndarray, shape (mesh.get_number_of_elements(),), optional
            Champ scalaire analytique associé à chaque élément.
            Si non précisé, la différence entre le champ numérique et analytique
            n'est pas calculée et affichée
        
        Returns
        -------
        None
        """
        
        plotter = MeshPlotter()

        if type(fieldA) == int:
            plot2 = 1
        else:
            plot2 = 2

        pl = pvQt.BackgroundPlotter(shape=(1, plot2)) 
        pl.show_grid() # Allow the execution to continue
        nodes, elements = plotter.prepare_data_for_pyvista(mesh)


        pv_mesh = pv.PolyData(nodes, elements)
        pv_mesh[title] = field
        pl.subplot(0, 0)
        pl.add_mesh(pv_mesh, show_edges=True, line_width=2, cmap="seismic",scalars=title)
        pl.camera_position = 'xy'

        if plot2 == 2:
            pv_mesh2 = pv.PolyData(nodes, elements)
            pv_mesh2["Erreur"] = fieldA-field
            pl.subplot(0, 1)
            pl.add_mesh(pv_mesh2, show_edges=True, line_width=2, cmap="seismic",scalars="Erreur")

        pl.camera_position = 'xy'
        pl.add_text(title, position="upper_edge")
        pl.show_grid()
        pl.show()



    @staticmethod
    def quadraticError(fieldA,fieldN):
        """
        Calcule l'erreur quadratique entre le champ numérique et analytique.
        
        Parameters
        ----------
        fieldA: ndarray, shape (mesh.get_number_of_elements(),)
            Champ scalaire analytique associé à chaque élément.
        fieldN: ndarray, shape (mesh.get_number_of_elements(),)
            Champ scalaire numérique associé à chaque élément.
        
        Returns
        -------
        D: float
            Erreur quadratique.
        """
        D = (fieldA-fieldN)**2
        D = np.mean(D)
        D = np.sqrt(D)
        return D
    

    
    @staticmethod
    def meanSize(Elements: list[Element]):
        """
        Calcule la taille moyenne des éléments.
        
        Parameters
        ----------
        Elements: list[Element]
            Liste des éléments.
        
        Returns
        -------
        float
            Taille moyenne des éléments.
        """
        Area = np.zeros(len(Elements))
        for e in Elements:
            Area[e._index] = e._area
        return np.mean(Area)**0.5
        
    @staticmethod
    def coupeY(mesh: Mesh,field,Y,plot : bool = True):

        """
        Performs a horizontal slice through a mesh at a specified Y-coordinate and 
        samples the values of a scalar field along the line.

        Parameters
        ----------
        mesh: Mesh
            The mesh object containing the data to be sliced.
        field: ndarray
            The scalar field associated with each element of the mesh.
        Y: float
            The Y-coordinate at which the slice is taken.
        plot: bool, optional
            If True, plots the sampled values along the slice. Default is True.

        Returns
        -------
        tuple[ndarray, ndarray]
            A tuple containing the distances along the X-axis and the corresponding 
            values of the scalar field at those distances.
        """
        plotter = MeshPlotter()
        nodes, elements = plotter.prepare_data_for_pyvista(mesh)

        pv_mesh = pv.PolyData(nodes, elements)
        pv_mesh["Champ T"] = field

        A = [pv_mesh.bounds[0],Y,0]
        B = [pv_mesh.bounds[1],Y,0]
        line_sample = pv_mesh.sample_over_line(A,B,resolution=100)

        values = line_sample["Champ T"]
        points = line_sample.points
        distance = points[:,0]

        if plot:
            plt.plot(distance,values)
            plt.title("T(X) (coupe en Y = {})".format(Y))
            plt.grid()
            plt.xlim(pv_mesh.bounds[0],pv_mesh.bounds[1])
            plt.show()

        return distance,values
    
    @staticmethod
    def coupeX(mesh: Mesh,field,X,plot : bool = True):

        """
        Renvoie les valeurs du champ scalaire le long d'une coupe X=cte de la mesh.

        Parameters
        ----------
        mesh : Mesh
            Le maillage contenant les données.
        field : ndarray
            Le champ scalaire associé à chaque élément du maillage.
        X : float
            La coordonnée X à laquelle la coupe est prise.
        plot : bool, optional
            Si True, trace les valeurs échantillonnées le long de la coupe. Par défaut, True.

        Returns
        -------
        tuple[ndarray, ndarray]
            Un tuple contenant les distances le long de l'axe Y et les valeurs du champ scalaire
            correspondantes à ces distances.
        """
        plotter = MeshPlotter()
        nodes, elements = plotter.prepare_data_for_pyvista(mesh)

        pv_mesh = pv.PolyData(nodes, elements)
        pv_mesh["Champ T"] = field

        A = [X,pv_mesh.bounds[2],0]
        B = [X,pv_mesh.bounds[3],0]
        line_sample = pv_mesh.sample_over_line(A,B,resolution=100)

        values = line_sample["Champ T"]
        points = line_sample.points
        distance = points[:,1]

        if plot:
            plt.plot(distance,values)
            plt.title("T(Y) (coupe en X = {})".format(X))
            plt.grid()
            plt.xlim(pv_mesh.bounds[0],pv_mesh.bounds[1])
            plt.show()

        return distance,values
    
    @staticmethod
    def getAnalyticalField(elements: list[Element],fct):
        """
        Calcule le champ scalaire analytique pour chaque élément.

        Parameters
        ----------
        elements: list[Element]
            Liste des éléments du maillage.
        fct: function
            Fonction prenant en entrée les coordonnées (x, y) et retournant
            la valeur analytique associée à ces coordonnées.

        Returns
        -------
        fieldA: ndarray
            Champ scalaire analytique associé à chaque élément du maillage.
        """
        fieldA = np.zeros(len(elements))

        for e in elements:
            Center = e._center
            fieldA[e._index] = fct(Center[0], Center[1])

        return fieldA
    

    @staticmethod
    def computeMMS(x : sp.Symbol,y : sp.Symbol ,fctSp,rho,k,Cp, velocityField : list):

        """
        Compute the analytical solution of a heat transfer problem using the method of manufactured solutions (MMS)

        Parameters
        ----------
        x : sympy symbol
            The x-coordinate
        y : sympy symbol
            The y-coordinate
        fctSp : sympy expression
            The analytical solution as a sympy expression
        rho : float
            The fluid density
        k : float
            The thermal conductivity
        Cp : float
            The heat capacity
        velocityField : list
            The velocity field [u,v]

        Returns
        -------
        fct : function
            The analytical solution as a function
        fgrad : function
            The gradient of the analytical solution as a function
        S : sympy expression
            The source term of the heat transfer equation
        """
        fct_x = fctSp.diff(x)
        fct_y = fctSp.diff(y)
        fct_xx = fct_x.diff(x)
        fct_yy = fct_y.diff(y)

        uftc_x = (velocityField[0]*fctSp).diff(x)
        uftc_y = (velocityField[1]*fctSp).diff(y)
        
        grad = [fct_x,fct_y]
        S = rho*Cp*(uftc_x + uftc_y) - k*(fct_xx + fct_yy)

        fct = sp.lambdify((x,y),fctSp,'numpy')
        fgrad = sp.lambdify((x,y),grad,'numpy')

        return fct,fgrad,S
    
    @staticmethod
    def pause():
        """
        Permet de suspendre l'exécution du programme en attendant que l'utilisateur appuie sur une touche.
        """
        
        input("Appuyez sur une touche pour continuer...")

    @staticmethod
    def plotVectorField(mesh: Mesh,field : np.ndarray, title : str,fieldName : str):
        """
        Affiche un champ vectoriel sur un maillage en utilisant le module pyvista.
        
        Parameters
        ----------
        mesh: Mesh
            Objet Mesh contenant les données du maillage.
        elements: list[Element]
            Liste des éléments du maillage.
        field: np.ndarray
            Champ vectoriel à afficher.
        title: str
            Titre de l'affichage.
        fieldName: str
            Nom du champ vectoriel à afficher.
            
        Returns
        -------
        None
        """
        plotter = MeshPlotter()
        nodes, elements = plotter.prepare_data_for_pyvista(mesh)

        pv_mesh = pv.PolyData(nodes, elements)
        pv_mesh[fieldName] = field

        pl = pvQt.BackgroundPlotter()
        pl.add_mesh(pv_mesh, scalars=fieldName, show_edges=True)

    
        Vectors = np.stack([field[:,0].flatten(), field[:,1].flatten(), np.zeros_like(field[:,0]).flatten()], axis=1)

        pl.add_arrows(pv_mesh.cell_centers().points, Vectors, mag=0.5)

        pl.add_text(title, position="upper_edge")
        pl.camera_position = 'xy'
        pl.show()

    @staticmethod
    def plotFaceVelocity(mesh: Mesh,field : np.ndarray,faceCenters: np.ndarray,FaceNormals : np.ndarray ,title : str):


        plotter = MeshPlotter()
        nodes, elements = plotter.prepare_data_for_pyvista(mesh)

        pv_mesh = pv.PolyData(nodes, elements)

        pl = pvQt.BackgroundPlotter()
        pl.add_mesh(pv_mesh, show_edges=True)


        field2D = np.zeros_like(FaceNormals)
        for i in range(len(FaceNormals)):
            field2D[i] = field[i] * FaceNormals[i]


        Centers = np.stack([faceCenters[:,0].flatten(), faceCenters[:,1].flatten(), np.zeros_like(faceCenters[:,0]).flatten()], axis=1)
        Vectors = np.stack([field2D[:,0].flatten(), field2D[:,1].flatten(), np.zeros_like(field2D[:,0]).flatten()], axis=1)

        pl.add_arrows(Centers, Vectors, mag=0.5)

        pl.add_text(title, position="upper_edge")
        pl.camera_position = 'xy'
        pl.show()






            




