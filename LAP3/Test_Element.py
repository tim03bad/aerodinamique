

#%%
from pyvista import Plotter
from element import Element
from mesh import Mesh
from meshGenerator import MeshGenerator
from meshConnectivity import MeshConnectivity
from meshPlotter import MeshPlotter
#%%

#%%
        
mesher = MeshGenerator()
plotter = MeshPlotter()
mesh_parameters = {'mesh_type': 'TRI','lc': 0.5}
mesh_obj = mesher.rectangle([0.0, 1.0, 0.0, 1.0], mesh_parameters)

conec = MeshConnectivity(mesh_obj)
conec.compute_connectivity()

plotter.plot_mesh(mesh_obj, label_points=True, label_elements=True, label_faces=True)
#%%
tri = Element(mesh_obj,0)
        #Triangles = [Element(mesh_obj) for i in range(mesh_obj.get_number_of_elements())] #Construction des elements

#%%


