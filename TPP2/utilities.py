import numpy as np
import pyvista as pv
import pyvistaqt as pvQt

from mesh import Mesh
from meshPlotter import MeshPlotter
from dataFace import dataFace
from element import Element

class Utilities:


    @staticmethod
    def FaceCoordinatesCalcultion(mesh_obj: Mesh,face : int,PolyList : list[Element]):

        """
        Calcul des coordonnes des sommets de la face et les centres des élements

        Parameters
        ----------
        mesh_obj : Mesh
            Objet contenant les informations du maillage
        face : int
            Numéro de la face
        PolyList : list[Element]
            Liste des  l ments du maillage

        Returns
        -------
        dataFace
            Objet contenant les coordonnées des sommets de la face et les centres des élements
        """

        FaceNodes = mesh_obj.get_face_to_nodes(face)
        Elems = mesh_obj.get_face_to_elements(face)
        Eg = PolyList[Elems[0]]
        Ed = PolyList[Elems[1]]
        
        Xa = mesh_obj.node_to_xcoord[FaceNodes[0]]
        Xb = mesh_obj.node_to_xcoord[FaceNodes[1]]

        Ya = mesh_obj.node_to_ycoord[FaceNodes[0]]
        Yb = mesh_obj.node_to_ycoord[FaceNodes[1]]

        XP = Eg.ElementCoord[0]
        XA = Ed.ElementCoord[0]

        YP = Eg.ElementCoord[1]
        YA = Ed.ElementCoord[1]

        return dataFace(Xa,Xb,Ya,Yb,XA,XP,YA,YP)
    
    @staticmethod
    def Pnksi(data : dataFace,DAi :float,dKsi : float):
        """
        Calcul de la quantité P_nksi

        Parameters
        ----------
        data : dataFace
            Objet contenant les coordonnées des sommets de la face et les centres des élements
        DAi : float
            Longueur de la face
        dKsi : float
            Longueur entre le centre de l'élément gauche et le centre de l'élément droit

        Returns
        -------
        float
            P_nksi
        """
        dXi = data.Xb - data.Xa
        dYi = data.Yb - data.Ya

        a = dYi*(data.XA-data.XP)/(DAi*dKsi)
        b = dXi*(data.YA-data.YP)/(DAi*dKsi)

        return a-b
    
    @staticmethod
    def Pksieta(data : dataFace,dKsi,DAi):

        """
        Calcul de la quantité P_ksieta

        Parameters
        ----------
        data : dataFace
            Objet contenant les coordonnées des sommets de la face et les centres des élements
        dKsi : float
            Longueur entre le centre de l'élément gauche et le centre de l'élément droit
        DAi : float
            Longueur de la face

        Returns
        -------
        float
            P_ksieta
        """
        a = (data.XA-data.XP)*(data.Xa-data.Xb)/(dKsi*DAi)
        b = (data.YA-data.YP)*(data.Ya-data.Yb)/(dKsi*DAi)

        return a+b 
    
    @staticmethod
    def Di(P_nksi : float,Gamma : float,DAi : float,dKsi : float):

        """
        Calcul de la diffusion normale

        Parameters
        ----------
        P_nksi : float
            Quantité P_nksi
        Gamma : float
            Coefficient de diffusion
        DAi : float
            Longueur de la face
        dKsi : float
            Longueur entre le centre de l'élément gauche et le centre de l'élément droit

        Returns
        -------
        float
            Quantité Di
        """
        return (1/P_nksi)*(Gamma*DAi)/dKsi
    
    @staticmethod
    def Sdcross(P_nksi : float,P_ksieta : float,Gamma : float,DAi : float, Eg : Element, Ed : Element,eta : np.ndarray, cD=True ):
    
        """
        Calcul du terme de diffusion croisée

        Parameters
        ----------
        P_nksi : float
            Quantité P_nksi
        P_ksieta : float
            Quantité P_ksieta
        Gamma : float
            Coefficient de diffusion
        DAi : float
            Longueur de la face
        Eg : Element
            El ment gauche
        Ed : Element
            El ment droit
        eta : ndarray
            Vecteur unitaire normal la face
        cD : boolean, optionnel
            Prise en compte ou non du coefficient de diffusion croisée

        Returns
        -------
        float
            Terme de diffusion crois e
        """
        Vect = (Eg.get_grad()+Ed.get_grad())/2

        if cD == True:
            return -Gamma*(P_ksieta/P_nksi)*(Vect@eta)*DAi
        else:
            return 0
    
    @staticmethod
    def pause():
        input("Appuyer sur une touche pour continuer...")

    @staticmethod
    def plot(mesh,fct,title:str,show_Mesh : bool = True,contour : bool = False):

        #Param d'affichage
        colorbar_args = {
            "title": "Champ T(K)",  # Titre de la colorbar
            "vertical": True,  # Colorbar verticale
            "position_x": 0.05,  # Position horizontale de la colorbar (proche du bord gauche)
            "position_y": 0.1,   # Position verticale, centrée
            "width": 0.05,       # Largeur de la colorbar
            "height": 0.8        # Hauteur de la colorbar
        }

        plotter = MeshPlotter()
        nodes,elements = plotter.prepare_data_for_pyvista(mesh)
        pv_mesh = pv.PolyData(nodes, elements)

        cell_centers = np.zeros((mesh.get_number_of_elements(), 2))
        for i_element in range(mesh.get_number_of_elements()):
            center_coords = np.array([0.0, 0.0])
            nodes = mesh.get_element_to_nodes(i_element)
            for node in nodes:
                center_coords[0] += mesh.get_node_to_xcoord(node)
                center_coords[1] += mesh.get_node_to_ycoord(node)
            center_coords /= nodes.shape[0]
            cell_centers[i_element, :] = center_coords


        pv_mesh['Champs T (analytique)'] = fct(cell_centers[:, 0], cell_centers[:, 1])

        pl = pvQt.BackgroundPlotter()
        pl.add_mesh(pv_mesh, scalars='Champs T (analytique)', show_edges=show_Mesh, cmap='hot',scalar_bar_args=colorbar_args)
        if contour:
                nodes_xcoords = mesh.get_nodes_to_xcoord()
                nodes_ycoords = mesh.get_nodes_to_ycoord()
                pv_mesh = pv_mesh.cell_data_to_point_data()
                contours = pv_mesh.contour(isosurfaces=15,scalars='Champs T (analytique)')
                pl.add_mesh(contours,color='k',show_scalar_bar=False,line_width=2)

        pl.camera_position = 'xy'
        pl.show_grid()
        pl.add_text(title, position="upper_edge")
        pl.show()

    @staticmethod
    def plotError(mesh,fctAnalytique,Values: np.ndarray,title:str,show_Mesh : bool = True):

     #Param d'affichage
        colorbar_args = {
            "title": "Champ T(K)",  # Titre de la colorbar
            "vertical": True,  # Colorbar verticale
            "position_x": 0.05,  # Position horizontale de la colorbar (proche du bord gauche)
            "position_y": 0.1,   # Position verticale, centrée
            "width": 0.05,       # Largeur de la colorbar
            "height": 0.8        # Hauteur de la colorbar
        }

        plotter = MeshPlotter()
        nodes,elements = plotter.prepare_data_for_pyvista(mesh)
        pv_mesh = pv.PolyData(nodes, elements)

        cell_centers = np.zeros((mesh.get_number_of_elements(), 2))
        for i_element in range(mesh.get_number_of_elements()):
            center_coords = np.array([0.0, 0.0])
            nodes = mesh.get_element_to_nodes(i_element)
            for node in nodes:
                center_coords[0] += mesh.get_node_to_xcoord(node)
                center_coords[1] += mesh.get_node_to_ycoord(node)
            center_coords /= nodes.shape[0]
            cell_centers[i_element, :] = center_coords


        pv_mesh['Champs T (analytique)'] = fctAnalytique(cell_centers[:, 0], cell_centers[:, 1]) - Values

        pl = pvQt.BackgroundPlotter()
        pl.add_mesh(pv_mesh, scalars='Champs T (analytique)', show_edges=show_Mesh, cmap='hot',scalar_bar_args=colorbar_args)
        pl.camera_position = 'xy'
        pl.show_grid()
        pl.add_text(title, position="upper_edge")
        pl.show()