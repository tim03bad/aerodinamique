#%% Importations
import numpy as np
import matplotlib.pyplot as plt

#%% Fonction solution analytique
def SolAnalytique(X:np.ndarray,N,L,F,D,Ta,Tb):
    
    P = (F*N)/D

    T = ((np.exp(P*X/L)-1)/(np.exp(P)-1))*(Tb-Ta) + Ta

    return T

#%% Calculs
def CalculSchemaCentre(N,L,F,D,Ta,Tb):

    ##Initialisation des matrices
    A = np.zeros((N,N)) #Matrice des coefficients
    b = np.zeros(N)     #Second membre (CL)

    ##Calculs préléminaires
    delta_x = L/N            #Longeur du volume de control
    X = np.arange(delta_x/2,L,delta_x) #Abssisses des volumes de control

    ## Conditions aux limites
    b[0] = (2*D+F)*Ta #Condition aux limites en A
    b[-1] = (2*D-F)*Tb #Condition aux limites en B

    #Construction de la matrice A
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

        pass

    ## Resolution du systeme
    T = np.linalg.solve(A,b)

    return X,T

def CalculUpwing(N,L,F,D,Ta,Tb):
    ##Initialisation des matrices
    A = np.zeros((N,N)) #Matrice des coefficients
    b = np.zeros(N)     #Second membre (CL)

    ##Calculs préléminaires
    delta_x = L/N            #Longeur du volume de control
    X = np.arange(delta_x/2,L,delta_x) #Abssisses des volumes de control

    #Remplissage de b
    b[0] = (2*D+F)*Ta
    b[-1] = (2*D)*Tb

    #Remplissage de la matrice A
    for i in range(N):
        if i == 0:
            A[i,i+1] = -(D + np.max([0,-F]))
            A[i,i] = 3*D+F
        elif i == N-1:
            A[i,i-1] = -(D + np.max([0,F]))
            A[i,i] = 3*D+F
        else:
            A[i,i-1] = -(D + np.max([0,F]))
            A[i,i+1] = -(D + np.max([0,-F]))
            A[i,i] = 2*D + np.max([0,F]) + np.max([0,-F])

        pass

        

    ## Resolution du systeme
    T = np.linalg.solve(A,b)

    return X,T



#%% Erreur et Ordre de convergence
def ErreurQuadratique(SolNum,SolAnalytique,pas):

    erreur = SolNum-SolAnalytique
    erreurCarre = erreur**2
    norme_erreur = np.sqrt(pas*np.sum(erreurCarre))
    
    return norme_erreur

def OrdreConvergenceCentre(N,L,F,Gamma,Ta,Tb):
    listOfN = [2,3,4,6,8,10,15,20,30, 50]
    E = []

    for n in listOfN:
        D = Gamma/(L/n)
        X,T = CalculSchemaCentre(n,L,F,D,Ta,Tb)
        ''' Debugging
        plt.plot(X,T, label = f'N = {n}')
        plt.plot(X,SolAnalytique(X,n,L,F,D,Ta,Tb),'x', label = f'Analytique {n}',)
        plt.legend()
        '''
        E.append(ErreurQuadratique(T,SolAnalytique(X,n,L,F,D,Ta,Tb),L/n))
    
    return np.array(listOfN), np.array(E), np.log(E[-2]/E[-1])/np.log((L/listOfN[-2])/(L/listOfN[-1]))

def OrdreConvergenceUpwing(N,L,F,Gamma,Ta,Tb):
    listOfN = [2,3,4,6,8,10,15,20,30, 50,100]
    E = []

    for n in listOfN:
        D = Gamma/(L/n)
        X,T = CalculUpwing(n,L,F,D,Ta,Tb)
        
        E.append(ErreurQuadratique(T,SolAnalytique(X,n,L,F,D,Ta,Tb),L/n))
    
    return np.array(listOfN), np.array(E), np.log(E[-2]/E[-1])/np.log((L/listOfN[-2])/(L/listOfN[-1]))


#%% Affichage résultat
def Graphique(Xnum,Xanalytique,SolNum,SolAnalytique,listOfN,E):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,8))

    #Graphique Solution numérique et analytique
    ax1.plot(Xnum,SolNum,'rx', label = 'Solution numérique')
    ax1.plot(Xanalytique,SolAnalytique, label = 'Solution analytique',)
    ax1.set(xlabel = 'Position', ylabel = 'Temperature', title = 'Solution numérique et analytique')
    ax1.legend()
    ax1.grid()

    #Graphique Convergence
    ax2.plot(np.log(1/listOfN),np.log(E), label = r'$\ln(E) = f(\ln(1/Nx))$')
    ax2.set(xlabel = 'log(1/N)', ylabel = 'log(E)', title = r'Courbe $\ln(E) = f(\ln(1/Nx))$')
    ax2.legend()
    ax2.grid()
    
    plt.tight_layout()
    plt.show()
    


#%% Exercice 5.1.i
#Paramètres
N = 5     #Nombre de volumes
u = 0.1   #Vitesse fluide (m/s)
rho = 1   #masse volumique (kg/m^3)
L = 1     #Longueur (m)
Gamma = 0.1 #Conductivite thermique

#CL 
Ta = 1
Tb = 0

#Calcul preliminaire
F = rho*u
D = Gamma/(L/N)

Xnum,Tnum = CalculSchemaCentre(N,L,F,D,Ta,Tb)
Xanalytique = np.linspace(0,L,100)
Tana = SolAnalytique(Xanalytique,N,L,F,D,Ta,Tb)
listOfN,E,p = OrdreConvergenceCentre(N,L,F,Gamma,Ta,Tb)
Graphique(Xnum,Xanalytique,Tnum,Tana,listOfN,E)

print("Exercice 5.1.i -> Ordre de convergence: {}".format(p))

#%% Exercice 5.1.ii
#Paramètres
N = 5     #Nombre de volumes
u = 2.5   #Vitesse fluide (m/s)
rho = 1   #masse volumique (kg/m^3)
L = 1     #Longueur (m)
Gamma = 0.1 #Conductivite thermique

#CL 
Ta = 1
Tb = 0

#Calcul preliminaire
F = rho*u
D = Gamma/(L/N)

Xnum,Tnum = CalculSchemaCentre(N,L,F,D,Ta,Tb)
Xanalytique = np.linspace(0,L,100)
Tana = SolAnalytique(Xanalytique,N,L,F,D,Ta,Tb)
listOfN,E,p = OrdreConvergenceCentre(N,L,F,Gamma,Ta,Tb)
Graphique(Xnum,Xanalytique,Tnum,Tana,listOfN,E)

print("Exercice 5.1.ii -> Ordre de convergence: {}".format(p))

#%% Exercice 5.1.ii
#Paramètres
N = 20    #Nombre de volumes
u = 2.5   #Vitesse fluide (m/s)
rho = 1   #masse volumique (kg/m^3)
L = 1     #Longueur (m)
Gamma = 0.1 #Conductivite thermique

#CL 
Ta = 1
Tb = 0

#Calcul preliminaire
F = rho*u
D = Gamma/(L/N)

Xnum,Tnum = CalculSchemaCentre(N,L,F,D,Ta,Tb)
Xanalytique = np.linspace(0,L,100)
Tana = SolAnalytique(Xanalytique,N,L,F,D,Ta,Tb)
listOfN,E,p = OrdreConvergenceCentre(N,L,F,Gamma,Ta,Tb)
Graphique(Xnum,Xanalytique,Tnum,Tana,listOfN,E)

print("Exercice 5.1.iii -> Ordre de convergence: {}".format(p))
# %% Exercice 5.2.i
#Paramètres
N = 5     #Nombre de volumes
u = 0.1   #Vitesse fluide (m/s)
rho = 1   #masse volumique (kg/m^3)
L = 1     #Longueur (m)
Gamma = 0.1 #Conductivite thermique

#CL 
Ta = 1
Tb = 0

#Calcul preliminaire
F = rho*u
D = Gamma/(L/N)

Xnum,Tnum = CalculUpwing(N,L,F,D,Ta,Tb)
Xanalytique = np.linspace(0,L,100)
Tana = SolAnalytique(Xanalytique,N,L,F,D,Ta,Tb)
listOfN,E,p = OrdreConvergenceUpwing(N,L,F,Gamma,Ta,Tb)
Graphique(Xnum,Xanalytique,Tnum,Tana,listOfN,E)

print("Exercice 5.2.i -> Ordre de convergence: {}".format(p))

#%% Exercice 5.2.ii
#Paramètres
N = 5     #Nombre de volumes
u = 2.5   #Vitesse fluide (m/s)
rho = 1   #masse volumique (kg/m^3)
L = 1     #Longueur (m)
Gamma = 0.1 #Conductivite thermique

#CL 
Ta = 1
Tb = 0

#Calcul preliminaire
F = rho*u
D = Gamma/(L/N)

Xnum,Tnum = CalculUpwing(N,L,F,D,Ta,Tb)
Xanalytique = np.linspace(0,L,100)
Tana = SolAnalytique(Xanalytique,N,L,F,D,Ta,Tb)
listOfN,E,p = OrdreConvergenceUpwing(N,L,F,Gamma,Ta,Tb)
Graphique(Xnum,Xanalytique,Tnum,Tana,listOfN,E)

print("Exercice 5.2.ii -> Ordre de convergence: {}".format(p))
# %%
