
from re import T
import unittest


from mesh import Mesh
import meshGenerator
from meshConnectivity import MeshConnectivity
from meshPlotter import MeshPlotter
import sympy as sp
import numpy as np
import pyvista as pv
import pyvistaqt as pvQt


class TestMesh(unittest.TestCase):

    def test_mesh(self):
        mesher = meshGenerator.MeshGenerator()
        plotter = MeshPlotter()

        mesh_parameters = {'mesh_type': 'TRI','lc': 0.5}
        number_of_holes = 0;

        mesh_obj = mesher.rectangle([0.0, 1.0, 0.0, 1.0], mesh_parameters)
        conec = MeshConnectivity(mesh_obj)
        conec.compute_connectivity()

        f = mesh_obj.get_number_of_elements();
        a = mesh_obj.get_number_of_faces();
        s = mesh_obj.get_number_of_nodes();

        print(f)
        print(a)
        print(s)


        self.assertTrue((f-a+s)==(1-number_of_holes))

       





        




if __name__ == '__main__' :
    unittest.main()

