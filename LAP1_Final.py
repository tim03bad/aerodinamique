'''
Emilien VIGIER
Timothée BADICHE
'''
#%%
import numpy as np
import matplotlib.pyplot as plt



#%% Fonctions solutions analytiques
def SolutionAnalytique4_1(x,L):        
        return x*400/L + 100

np.vectorize(SolutionAnalytique4_1)

def SolutionAnalytique4_2(x,L,k,q,Ta,Tb):
        a = (Tb-Ta)/L + q*(L-x)/(2*k)
        return a*x+Ta

np.vectorize(SolutionAnalytique4_2)

def SolutionAnalytique4_3(x : np.ndarray,L,n2,Tb,Tinf):
        n = np.sqrt(n2)
        
        frac = np.cosh(n*(L-x))/np.cosh(n*L)
        
        return frac*(Tb-Tinf) + Tinf


#%% Calculs
def Calcul4_1(N,L,k,S,Ta,Tb):
    
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

def Calcul4_2(N,L,k,q,S,Ta,Tb):
    
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
    b = b*S_u                            #Ajout du terme source uniforme
    b[0] = b[0] + (2*k*S/delta_x)*Ta     #Condition aux limites en A
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

def Calcul4_3(N,L,n2,Tb,Tinf):

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
    b = b*S_u              #Ajout terme source uniforme au second membre
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
    Tnum = np.linalg.solve(A, b)
    
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

    #Calcul de l'erreur quadratique pour une série de maillage de taille différente
    for n in listOfN:
        X,T = Calcul4_1(n,L,k,S,Ta,Tb)
        E.append(ErreurQuadratique(T,SolutionAnalytique4_1(X,L),L/n))
    
    return np.array(listOfN), np.array(E), np.log(E[0]/E[1])/np.log((L/listOfN[0])/(L/listOfN[1]))

def OrdeConvergence4_2(Ta,Tb,q,k,S,L):
    listOfN = [2,3,4,6,8,10,15,20,30,50,100]
    E = []
    
    #Calcul de l'erreur quadratique pour une série de maillage de taille différente   
    for n in listOfN:
        X,T = Calcul4_2(n,L,k,q,S,Ta,Tb)
        E.append(ErreurQuadratique(T,SolutionAnalytique4_2(X,L,k,q,Ta,Tb),L/n))
    
    return  np.array(listOfN) ,np.array(E), np.log(E[0]/E[1])/np.log((L/listOfN[0])/(L/listOfN[1]))

def OrdeConvergence4_3(Tb,Tinf,L,n2):
    listOfN = [2,3,4,6,8,10,15,20,30,50,100]
    E = []

    #Calcul de l'erreur quadratique pour une série de maillage de taille différente
    for n in listOfN:
        X,T = Calcul4_3(n,L,n2,Tb,Tinf)
        E.append(ErreurQuadratique(T,SolutionAnalytique4_3(X,L,n2,Tb,Tinf),L/n))
    
    return np.array(listOfN),np.array(E), np.log(E[-2]/E[-1])/np.log((L/listOfN[-2])/(L/listOfN[-1]))

#%% Graphiques
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


#%% Exercice 4.1
L = 0.5 #Longeur de la barre
N = 5  #Nombre de noeud
k = 1000  #Conductivité thermique
S = 10e-3 #Cross-section
Ta = 100  #Température au bord A
Tb = 500  # Température au bord B

Xnum,Tnum = Calcul4_1(N,L,k,S,Ta,Tb)
Xanalytique = np.linspace(0,L,num=50)
SolAnalytique = SolutionAnalytique4_1(Xanalytique,L)
listOfN, E, p = OrdeConvergence4_1(Ta,Tb,k,S,L)
Graphique(Xnum,Xanalytique,Tnum,SolAnalytique,listOfN,E)

print("Exercice 4.1 -> Ordre de convergence: {}".format(p))

#%% Exercice 4.2
L = 0.02 #Longueur de la barre
N = 10  #Nombre de noeud
k = 0.5  #Conductivité thermique
S = 1 #Cross-section
q = 1000e3  #Source de chaleur
Ta = 100   #Temparature au bord A
Tb = 200   #Temperature au bord B

Xnum,Tnum = Calcul4_2(N,L,k,q,S,Ta,Tb)
Xanalytique = np.linspace(0,L,num=50)
SolAnalytique = SolutionAnalytique4_2(Xanalytique,L,k,q,Ta,Tb)
listOfN, E, p = OrdeConvergence4_2(L,k,q,S,Ta,Tb)
Graphique(Xnum,Xanalytique,Tnum,SolAnalytique,listOfN,E)

print("Exercice 4.2 -> Ordre de convergence: {}".format(p))


#%% Exercice 4.3
L = 1        #Longueur de la l'ailette
N = 50        #Nombre de noeud
n2 = 25       #Coefficient réduit
Tb = 100      #Température au bord B (thermostat)
Tinf = 20     #Température de l'air ambiant à l'infini

Xnum,Tnum = Calcul4_3(N,L,n2,Tb,Tinf)
Xanalytique = np.linspace(0,L,num=50)
SolAnalytique = SolutionAnalytique4_3(Xanalytique,L,n2,Tb,Tinf)
listOfN, E, p = OrdeConvergence4_3(Tb,Tinf,L,n2)
Graphique(Xnum,Xanalytique,Tnum,SolAnalytique,listOfN,E)

print("Exercice 4.3 -> Ordre de convergence: {}".format(p))


# %%
