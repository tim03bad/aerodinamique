#%%
from re import T
import sympy as sp
import numpy as np


#%%
x, y = sp.symbols('x y')

#%%
u = (2*x**2 - x**4 -1)*(y-y**3)
v = -(2*y**2 - y**4 -1)*(x-x**3)
fu = sp.lambdify((x, y), u, 'numpy')
fv = sp.lambdify((x, y), v, 'numpy')

#%%
T0 = 400
Tx = 50
Txy = 100
Tmms = T0 + Tx*sp.cos(sp.pi*x) + Txy*sp.sin(sp.pi*x*y)
TmmsGrad = [sp.diff(Tmms, x), sp.diff(Tmms, y)]

fTGrad = sp.lambdify((x, y), TmmsGrad, 'numpy')


#%%
rho = 1
Cp = 1
k = 1

#%%
uTmms_x = sp.diff(Tmms*u, x)
vTmms_y = sp.diff(Tmms*v, y)

Tmms_xx = sp.diff(Tmms, x, 2)
Tmms_yy = sp.diff(Tmms, y, 2)
#%%
S = -k*(Tmms_xx + Tmms_yy) + rho*Cp*(uTmms_x + vTmms_y)

#%%


fT = sp.lambdify((x, y), Tmms, 'numpy')

#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
xi = np.linspace(-1, 1, 100)
yi = np.linspace(-1, 1, 100)
Xi, Yi = np.meshgrid(xi, yi)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Xi, Yi, fv(Xi, Yi), cmap='viridis')
plt.show()

# %%
