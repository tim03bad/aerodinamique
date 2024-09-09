
#%% Importations
import numpy as np
import matplotlib.pyplot as plt

#%% Parametres
N = 5     #Nombre de volumes
u = 0.1   #Vitesse fluide (m/s)
rho = 1   #masse volumique (kg/m^3)
L = 1     #Longueur (m)
Gamma = 0.1 #Conductivite thermique

PhiA = 1
PhiB = 0

#%% Calculs preliminaires
delta_x = L/N

F = rho*u   #Convective mass flux
D = Gamma/delta_x  #Diffusion conductance at cell faces

#%% Matrices
A = np.zeros((N,N))
b = np.zeros(N)


#Remplissage de la matrice A
for i in range(N):
    if i == 0:
        A[i,i+1] = F/2-D
        A[i,i] = 3*D+F/2
    elif i == N-1:
        A[i,i-1] = -(D+F/2)
        A[i,i] = 3*D-F/2
    else:
        A[i,i-1] = -(D+F/2)
        A[i,i+1] = -(D-F/2)
        A[i,i] = 2*D

#Remplissage de b
b[0] = (2*D+F)*PhiA
b[-1] = (2*D-F)*PhiB

# %% Résolution du problème
Phi = np.linalg.solve(A,b)
# %%
def SolAnalytique(x):
    return (2.7183-np.exp(x))/1.7183
# %%
Xnum = np.arange(delta_x/2,L,delta_x)
Xanalytique = np.linspace(0,L,num=50)

PhiAnalytique = SolAnalytique(Xanalytique)

plt.plot(Xnum,Phi,'rx')
plt.plot(Xanalytique,PhiAnalytique)
plt.xlim(0, L)
plt.plot()

plt.show()
# %%
