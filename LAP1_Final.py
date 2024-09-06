
#%%
import numpy as np
import matplotlib.pyplot as plt


#%% Paramètres
L = 0.02    #Longeur de la barre
N = 10      #Nombre de noeud
k = 0.5     #Conductivité thermique
S = 1       #Cross-section
q = 1000e3  # Conductivité thermique

#Conditions aux limites
Ta = 100
Tb = 200
Tinf = 25   #Temperature air ambiant

#%% Fonctions solutions analytiques
def SolutionAnalytique4_1(x,L):        
        return x*400/L + 100

np.vectorize(SolutionAnalytique4_1)

def SolutionAnalytique4_2(x,L,Ta,Tb,k,q):
        a = (Tb-Ta)/L + q*(L-x)/(2*k)
        return a*x+Ta

np.vectorize(SolutionAnalytique4_2)

def SolutionAnalytique4_3(x,L,Tb,n2,Tinf):
        n = np.sqrt(n2)
        
        frac = np.cosh(n*(L-x))/np.cosh(n*L)
        
        return frac*(Tb-Tinf) + Tinf

np.vectorize(SolutionAnalytique4_3)

#%% Calculs
def Calcul4_1(N,Ta,Tb,k,S,L):
    
    ##Initialisation des matrices
    A = np.zeros((N,N)) #Matrice des coefficients
    b = np.zeros(N)     #Second membres (CL)

    ##Calculs préléminaires
    delta_x = L/N            #Longeur du volume de control
    X = np.arange(delta_x/2,L,delta_x) #Abssisses des volumes de control


    ## Conditions aux limites

    b[0] = (2*k*S/delta_x)*Ta #Condition aux limites en A
    b[-1] = ((2*k*S)/delta_x)*Tb #Condition aux limites en B

    #Construction de la matrice A
    for i in range(0, N):
        
        if i == 0: #Frontière A
            A[i,i+1]= -(k*S)/delta_x
            A[i,i] = 3*(k*S)/delta_x
            
        elif i == N-1: #Frontière B
            A[i,i-1] = -(k*S)/delta_x
            A[i,i] = 3*(k*S)/delta_x
            
        else:
            A[i,i-1] = -k*S/delta_x
            A[i,i+1] = -k*S/delta_x
            A[i,i]   = 2*k*S/delta_x
        
        pass

    ##Resolution 
    Tnum = np.linalg.solve(A, b)
    
    return X,Tnum

def Calcul4_2(N,Ta,Tb,q,k,S,L):
    
    ##Initialisation des matrices
    A = np.zeros((N,N)) #Matrice des coefficients
    b = np.ones(N) #Second membres (CL)

    ##Calculs préléminaires
    delta_x = L/N #Longeur du volume de control
    X = np.arange(delta_x/2,L,delta_x) #Abssisses des volumes de control
    
    a_w = k*S/delta_x  #i-1
    a_e = k*S/delta_x  #i+1
    a_p = a_w+a_e      #i
    S_u = q*S*delta_x  #Terme source uniforme
    
    ## Conditions aux limites et Sources
    b = b*S_u
    b[0] = b[0] + (2*k*S/delta_x)*Ta #Condition aux limites en A
    b[-1] = b[-1] + ((2*k*S)/delta_x)*Tb #Condition aux limites en B

    #Construction de la matrice A
    for i in range(0, N):
        
        if i == 0: #Frontière A
            A[i,i+1]= -a_e
            A[i,i] = 3*k*S/delta_x #correction avec CL
            
        elif i == N-1: #Frontière B
            A[i,i-1] = -a_w
            A[i,i] = 3*k*S/delta_x #correction avec CL
        else:
            A[i,i-1] = -a_w
            A[i,i+1] = -a_e
            A[i,i] = a_p
        
        pass

    ##Resolution 
    Tnum = np.linalg.solve(A, b)
    
    return X,Tnum

def Calcul4_3(N,Tb,Tinf,L,n2):

    ##Initialisation des matrices
    A = np.zeros((N,N)) #Matrice des coefficients
    b = np.ones(N)      #Second membres (CL)

    ##Calculs préléminaires
    delta_x = L/N #Longeur du volume de control
    X = np.arange(delta_x/2,L,delta_x) #Abssisses des volumes de control
    
    a_w = 1/delta_x #i-1
    a_e = 1/delta_x #i+1
    a_p = a_w + a_e + n2*delta_x  #i
    S_u = n2*delta_x*Tinf  #Terme source uniforme
    
    ## Conditions aux limites et Sources
    b = b*S_u
    b[0] += (2/delta_x)*Tb #Condition aux limites en A

    #Construction de la matrice A
    for i in range(0, N):
        
        if i == 0: #Frontière A
            A[i,i+1]= -a_e
            A[i,i] = 1/delta_x + n2*delta_x + 2/delta_x #correction avec CL
            
        elif i == N-1: #Frontière B
            A[i,i-1] = -a_w
            A[i,i] = 1/delta_x + n2*delta_x #correction avec CL
        else:
            A[i,i-1] = -a_w
            A[i,i+1] = -a_e
            A[i,i] = a_p
        
        pass

    ##Resolution 
    X,Tnum = np.linalg.solve(A, b)
    
    return X,Tnum

#%% Erreur et Ordre de convergence
def ErreurQuadratique(SolNum,SolAnalytique,pas):

    erreur = SolNum-SolAnalytique
    erreurCarre = erreur**2
    norme_erreur = np.sqrt(pas*np.sum(erreurCarre))
    
    return norme_erreur

def OrdeConvergence4_1(Ta,Tb,k,S,L):
    listOfN = [2,3,4,6,8,10,15,20,30, 50,100]
    E = []
    
    for n in listOfN:
        X,T = Calcul4_1(n,Ta,Tb,k,S,L)
        E.append(ErreurQuadratique(T,SolutionAnalytique4_1(X),L/n))
    
    return np.array(E), np.log(E[0]/E[1])/np.log((L/listOfN[0])/(L/listOfN[1]))

def OrdeConvergence4_2(Ta,Tb,q,k,S,L):
    listOfN = [2,3,4,6,8,10,15,20,30,50,100]
    E = []
    
    for n in listOfN:
        X,T = Calcul4_2(n,Ta,Tb,q,k,S,L)
        E.append(ErreurQuadratique(T,SolutionAnalytique4_2(X),L/n))
    
    return  np.array(E), np.log(E[0]/E[1])/np.log((L/listOfN[0])/(L/listOfN[1]))

def OrdeConvergence4_3(Tb,Tinf,L,n2):
    listOfN = [2,3,4,6,8,10,15,20,30,50,100]
    E = []
    
    for n in listOfN:
        X,T = Calcul4_3(n,Tb,Tinf,L,n2)
        E.append(ErreurQuadratique(T,SolutionAnalytique4_3(X),L/n))
    
    return np.array(E), np.log(E[0]/E[1])/np.log((L/listOfN[0])/(L/listOfN[1]))

#%% Graphiques