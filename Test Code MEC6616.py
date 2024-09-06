# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 21:22:28 2024

@author: GU603ZW
"""
#%%
import numpy as np
import matplotlib.pyplot as plt

def ErreurQuadratique(SolNum,SolAnalytique,pas):
    erreur = SolNum-SolAnalytique
    erreurCarre = erreur**2
    norme_erreur = np.sqrt(pas*np.sum(erreurCarre))
    
    return norme_erreur

def Calcul4_1(N,Ta,Tb,k,S,L):
    """
    

    Parameters
    ----------
    N : Int
        Nombre de noeuds.
    Ta : Int/float
        Température au bord au point A.
    Tb : Int/float
        Température au bord au point B.
    k : Int/float
        Conductivité thermique
    S : Int/float
        Surface transversale
    L : Int/float
        Longeur

    Returns
    -------
    Tnum : np.Array
        Vecteur solution contenant la température aux noeuds.

    """
    
    ##Initialisation des matrices
    A = np.zeros((N,N)) #Matrice des coefficients
    b = np.zeros(N) #Second membres (CL)

    ##Calculs préléminaires
    delta_x = L/N #Longeur du volume de control


    ## Conditions aux limites

    b[0] = (2*k*S/delta_x)*Ta #Condition aux limites en A
    b[-1] = ((2*k*S)/delta_x)*Tb #Condition aux limites en B

    #Construction de la matrice des coeffs
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
            A[i,i] = 2*k*S/delta_x
        
        pass

    ##Resolution 
    Tnum = np.linalg.solve(A, b)
    
    return Tnum
    
def Calcul4_2(N,Ta,Tb,q,k,S,L):

    ##Initialisation des matrices
    A = np.zeros((N,N)) #Matrice des coefficients
    b = np.ones(N) #Second membres (CL)

    ##Calculs préléminaires
    delta_x = L/N #Longeur du volume de control
    
    a_w = k*S/delta_x #i-1
    a_e = k*S/delta_x #i+1
    a_p = a_w+a_e  #i
    S_u = q*S*delta_x
    
    ## Conditions aux limites et Sources
    b = b*S_u
    b[0] = b[0] + (2*k*S/delta_x)*Ta #Condition aux limites en A
    b[-1] = b[-1] + ((2*k*S)/delta_x)*Tb #Condition aux limites en B

    #Construction de la matrice des coeffs
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
    
    return Tnum

def Calcul4_3(N,Tb,Tinf,L,n2):

    ##Initialisation des matrices
    A = np.zeros((N,N)) #Matrice des coefficients
    b = np.ones(N) #Second membres (CL)

    ##Calculs préléminaires
    delta_x = L/N #Longeur du volume de control
    
    a_w = 1/delta_x #i-1
    a_e = 1/delta_x #i+1
    a_p = a_w + a_e + n2*delta_x  #i
    S_u = n2*delta_x*Tinf
    
    ## Conditions aux limites et Sources
    b = b*S_u
    b[0] = b[0] + (2/delta_x)*Tb #Condition aux limites en A

    #Construction de la matrice des coeffs
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
    
    return Tnum


def Resolution4_1():
    ##Solution Analytique
    def SolutionAnalytique4_1(x):
        return x*400/L + 100
     
    np.vectorize(SolutionAnalytique4_1)
    
    ##Définition des paramètres
    L = 0.5 #Longeur de la barre
    N = 5  #Nombre de noeud
    k = 1000  #Conductivité thermique
    S = 10e-3 #Cross-section

    ##Condition aux limites
    Ta = 100
    Tb = 500

    ##Calculs préliminaires
    delta_x = L/N

    ##Resolution 
    Tnum = Calcul4_1(N, Ta, Tb,k,S,L)

    ## Post traitement et solution analytique
    X = np.arange(delta_x/2,L,delta_x) # Position des points discrets
    Xanalytique = np.linspace(0,L,num=50) #Abscisses de calculs solution analytique 




    ##Tracés
    plt.figure(figsize=(6,8))

    plt.subplot(2,1,1) #Graphique Courbe Temperature
    plt.plot(X, Tnum,'rx',label='Solution numérique')
    plt.plot(Xanalytique,SolutionAnalytique4_1(Xanalytique),'-',label='Solution Analytique')
    plt.xlim(0,L)
    plt.legend()
    plt.grid()
    plt.title("Solutions numérique et analytique")

    plt.subplot(2,1,2) # Graphique Evolution précision numérique

    ## Evolution de l'erreur quadratique en fct du nombre de points
    listOfN = [2,3,4,6,8,10,15,20,30,50,100]
    E = []

    for n in listOfN:
        delta_x = L/n
        Xt = np.arange(delta_x/2,L,delta_x)
        
        T = Calcul4_1(n, Ta, Tb,k,S,L)
        
        E.append(ErreurQuadratique(T,SolutionAnalytique4_1(Xt), delta_x))

    listOfN = np.array(listOfN)
    E = np.array(E)
    
    
    plt.plot(np.log(1/listOfN),np.log(E),'*-')
    plt.title(r'Courbe $\ln(E) = f(\ln(1/Nx))$')
    plt.grid();
     
    ## Calcul de l'ordre de convergence
    p = np.log(E[-2]/E[-1])/np.log((L/listOfN[-2])/(L/listOfN[-1]))
    print("Ordre de convergence: {}".format(p))
    
    return

def Resolution4_2():
    ##Solution Analytique
    def SolutionAnalytique4_2(x):
        a = (Tb-Ta)/L + q*(L-x)/(2*k)
        return a*x+Ta
    np.vectorize(SolutionAnalytique4_2)
    
    ##Définition des paramètres
    L = 0.02 #Longeur de la barre
    N = 10  #Nombre de noeud
    k = 0.5  #Conductivité thermique
    S = 1 #Cross-section
    q = 1000e3

    ##Condition aux limites
    Ta = 100
    Tb = 200

    ##Calculs préliminaires
    delta_x = L/N

    ##Resolution 
    Tnum = Calcul4_2(N, Ta, Tb,q,k,S,L)

    ## Post traitement et solution analytique
    X = np.arange(delta_x/2,L,delta_x) # Position des points discrets
    Xanalytique = np.linspace(0,L,num=50) #Abscisses de calculs solution analytique 




    ##Tracés
    plt.figure(figsize=(6,8))

    plt.subplot(2,1,1) #Graphique Courbe Temperature
    plt.plot(X, Tnum,'rx',label='Solution numérique')
    plt.plot(Xanalytique,SolutionAnalytique4_2(Xanalytique),'-',label='Solution Analytique')
    plt.xlim(0,L)
    plt.legend()
    plt.grid()
    plt.title("Solutions numérique et analytique")

    plt.subplot(2,1,2) # Graphique Evolution précision numérique

    ## Evolution de l'erreur quadratique en fct du nombre de points
    listOfN = [2,3,4,6,8,10,15,20,30,50,100]
    E = []

    for n in listOfN:
        delta_x = L/n
        Xt = np.arange(delta_x/2,L,delta_x)
        
        T = Calcul4_2(n, Ta, Tb,q,k,S,L)
        
        E.append(ErreurQuadratique(T,SolutionAnalytique4_2(Xt), delta_x))
        

    listOfN = np.array(listOfN)
    E = np.array(E)

    plt.plot(np.log(1/listOfN),np.log(E),'*-')
    plt.title(r'Courbe $\ln(E) = f(\ln(1/Nx))$')
    plt.grid();
     
    ## Calcul de l'ordre de convergence
    p = np.log(E[-2]/E[-1])/np.log((L/listOfN[-2])/(L/listOfN[-1]))
    print("Ordre de convergence: {}".format(p))
    
    return

 
def Resolution4_3():
    ##Solution Analytique
    def SolutionAnalytique4_3(x):
        n = np.sqrt(n2)
        
        frac = np.cosh(n*(L-x))/np.cosh(n*L)
        
        return frac*(Tb-Tinf) + Tinf
    
    np.vectorize(SolutionAnalytique4_3)
    
    ##Définition des paramètres
    N = 50
    L = 1
    n2 = 25
    
    Tb = 100
    Tinf = 20
    
    
    
    ##calculs préliminaire
    delta_x = L/N
    
    ## Résolution
    Tnum = Calcul4_3(N, Tb, Tinf, L, n2)

    ## Post traitement et solution analytique
    X = np.arange(delta_x/2,L,delta_x) # Position des points discrets
    Xanalytique = np.linspace(0,L,num=50) #Abscisses de calculs solution analytique 




    ##Tracés
    plt.figure(figsize=(6,8))

    plt.subplot(2,1,1) #Graphique Courbe Temperature
    plt.plot(X, Tnum,'rx',label='Solution numérique')
    plt.plot(Xanalytique,SolutionAnalytique4_3(Xanalytique),'-',label='Solution Analytique')
    plt.xlim(0,L)
    plt.legend()
    plt.grid()
    plt.title("Solutions numérique et analytique")

    plt.subplot(2,1,2) # Graphique Evolution précision numérique

    ## Evolution de l'erreur quadratique en fct du nombre de points
    listOfN = [2,3,4,6,8,10,15,20,30,50,100]
    E = []

    for n in listOfN:
        delta_x = L/n
        Xt = np.arange(delta_x/2,L,delta_x)
        
        T = Calcul4_3(n, Tb, Tinf, L, n2)
        
        E.append(ErreurQuadratique(T,SolutionAnalytique4_3(Xt), delta_x))

    listOfN = np.array(listOfN)
    E = np.array(E)

    plt.plot(np.log(1/listOfN),np.log(E),'*-')
    plt.title(r'Courbe $\ln(E) = f(\ln(1/Nx))$')
    plt.grid();
    
    ## Calcul de l'ordre de convergence
    p = np.log(E[-2]/E[-1])/np.log((L/listOfN[-2])/(L/listOfN[-1]))
    print("Ordre de convergence: {}".format(p))
    return

Resolution4_3()