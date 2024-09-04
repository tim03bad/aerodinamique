import numpy as np
import matplotlib.pyplot as plt
nb_points_max=100

'''
k : conductivité thermique
A : aire de la face que traverse le flux thermique
L : longueur
nb_points : nombre de points de discrétisation
Ta : Température en 0
Tb : Température en deltaX*nb_points
p : Périmètre de A
h : coefficient de transfert thermique
q : génération de chaleur par volume par seconde
T_ambient : Température ambiante
'''

'''création de la matrice des valeurs de Su (matrice B du système AX=B) pour l'exemple 1'''
def matrice_temp_1(k,A,L,nb_points,Ta,Tb) :
    deltaX=L/nb_points
    c=k*A/deltaX
    M=np.zeros(nb_points)
    M[0]=2*Ta*c
    M[-1]=2*Tb*c
    return M

'''création de la matrice des valeurs -aw ap -ae (matrice A du système AX=B) pour l'exemple 1'''
def matrice_equation_1(k,A,L,nb_points):
    deltaX=L/nb_points
    c=k*A/deltaX
    M=np.zeros((nb_points,nb_points))
    M[0][0]=3*c
    M[0][1]=-c
    M[-1][-1]=3*c
    M[-1][-2]=-c
    for i in range(1,nb_points-1):
        M[i][i-1]=-c
        M[i][i]=2*c
        M[i][i+1]=-c
    return M

'''création de la matrice des valeurs de Su (matrice B du système AX=B) pour l'exemple 2'''
def matrice_temp_2(k,A,L,nb_points,Ta,Tb,q) :
    deltaX=L/nb_points
    c=k*A/deltaX
    M=np.zeros(nb_points)
    M[0]=2*Ta*c + q*A*deltaX
    M[-1]=2*Tb*c + q*A*deltaX
    for i in range(1,nb_points-1):
        M[i]=q*A*deltaX
    return M

'''création de la matrice des valeurs -aw ap -ae (matrice A du système AX=B) pour l'exemple 2'''
def matrice_equation_2(k,A,L,nb_points):
    deltaX=L/nb_points
    c=k*A/deltaX
    M=np.zeros((nb_points,nb_points))
    M[0][0]=3*c
    M[0][1]=-c
    M[-1][-1]=3*c
    M[-1][-2]=-c
    for i in range(1,nb_points-1):
        M[i][i-1]=-c
        M[i][i]=2*c
        M[i][i+1]=-c
    return M

'''création de la matrice des valeurs de Su (matrice B du système AX=B) pour l'exemple 3'''
def matrice_temp_3(k,A,L,nb_points,Ta,p,h,T_ambient) :
    deltaX=L/nb_points
    n2=h*p/(k*A)
    M=np.zeros(nb_points)
    M[0]=2*Ta/deltaX + deltaX*T_ambient*n2
    M[-1]=deltaX*T_ambient*n2
    for i in range(1,nb_points-1):
        M[i]=deltaX*T_ambient*n2
    return M



'''création de la matrice des valeurs -aw ap -ae (matrice A du système AX=B) pour l'exemple 3'''
def matrice_equation_3(k,A,L,nb_points,p,h):
    deltaX=L/nb_points
    M=np.zeros((nb_points,nb_points))
    n2=h*p/(k*A)
    M[0][0]=3/deltaX+n2*deltaX
    M[0][1]=-1/deltaX
    M[-1][-1]=1/deltaX+n2*deltaX
    M[-1][-2]=-1/deltaX
    for i in range(1,nb_points-1):
        M[i][i-1]=-1/deltaX
        M[i][i]=2/deltaX+n2*deltaX
        M[i][i+1]=-1/deltaX
    return M

'''résolution du système, retourne un vecteur de chaleur (matrice X du système AX=B) pour l'exemple 1'''
def solution_1(k,A,L,nb_points,Ta,Tb):
    a=matrice_equation_1(k,A,L,nb_points)
    b=matrice_temp_1(k,A,L,nb_points,Ta,Tb)
    return np.linalg.solve(a, b)

'''résolution du système, retourne un vecteur de chaleur (matrice X du système AX=B) pour l'exemple 2'''
def solution_2(k,A,L,nb_points,Ta,Tb,q):
    a=matrice_equation_2(k,A,L,nb_points)
    b=matrice_temp_2(k,A,L,nb_points,Ta,Tb,q)
    return np.linalg.solve(a, b)

'''résolution du système, retourne un vecteur de chaleur (matrice X du système AX=B) pour l'exemple 3'''
def solution_3(k,A,L,nb_points,Ta,p,h,T_ambient):
    a=matrice_equation_3(k,A,L,nb_points,p,h)
    b=matrice_temp_3(k,A,L,nb_points,Ta,p,h,T_ambient)
    return np.linalg.solve(a, b)

'''solution analytique de l'exemple 1'''
def solution_analytique_1(x,L,nb_points,Ta,Tb):
    return Ta + x*(Tb-Ta)/L

'''solution analytique de l'exemple 2'''
def solution_analytique_2(x,L,nb_points,Ta,Tb,k,q):
    return ((Tb-Ta)/L+(L-x)*q/(2*k))*x+Ta

'''solution analytique de l'exemple 3'''
def solution_analytique_3(x,nb_points,Ta,p,h,T_ambient,k,A):
    n2=h*p/(k*A)
    return (Ta-T_ambient)*np.exp(-x*np.sqrt(n2))+T_ambient

'''retourne la liste des abscisses, de la solution analytique et de la solution numérique de l'exemple 1'''
def list_graph_1(k,A,L,nb_points,Ta,Tb): 
    deltaX=L/nb_points
    X=[0] + [(i+1/2)*deltaX for i in range(nb_points)]
    X.append(nb_points*deltaX)
    Y_analytique=[solution_analytique_1(x,L,nb_points,Ta,Tb) for x in X]
    Y_numerique=solution_1(k,A,L,nb_points,Ta,Tb)
    Y_numerique=np.insert(Y_numerique, 0, Ta)
    Y_numerique=np.append(Y_numerique, Tb)
    return X,Y_numerique,Y_analytique
    
'''retourne la liste des abscisses, de la solution analytique et de la solution numérique de l'exemple 2'''
def list_graph_2(k,A,L,nb_points,Ta,Tb,q): 
    deltaX=L/nb_points
    X=[0] + [(i+1/2)*deltaX for i in range(nb_points)]
    X.append(nb_points*deltaX)
    Y_analytique=[solution_analytique_2(x,L,nb_points,Ta,Tb,k,q) for x in X]
    Y_numerique=solution_2(k,A,L,nb_points,Ta,Tb,q)
    Y_numerique=np.insert(Y_numerique, 0, Ta)
    Y_numerique=np.append(Y_numerique, Tb)
    return X,Y_numerique,Y_analytique
    
'''retourne la liste des abscisses, de la solution analytique et de la solution numérique de l'exemple 3'''
def list_graph_3(k,A,L,nb_points,Ta,p,h,T_ambient): 
    deltaX=L/nb_points
    X=[0] + [(i+1/2)*deltaX for i in range(nb_points)]
    Y_analytique=[solution_analytique_3(x,nb_points,Ta,p,h,T_ambient,k,A) for x in X]
    Y_numerique=solution_3(k,A,L,nb_points,Ta,p,h,T_ambient)
    Y_numerique=np.insert(Y_numerique, 0, Ta)
    return X,Y_numerique,Y_analytique

'''graph de comparaison entre la solution numérique et la solution analytique'''
def graph(X,Y_numerique,Y_analytique):
    plt.plot(X,Y_numerique,'o')
    plt.plot(X,Y_analytique)
    plt.ylim(0, max(max(Y_numerique), max(Y_analytique))*1.1)
    plt.show()
    
'''fait la moyenne de la norme 1 d'erreur pour tout les points de la liste'''
def erreur_L1(Y_numerique,Y_analytique):
    erreur=0
    n=len(Y_numerique)
    for i in range(n):
        erreur+=abs(Y_numerique[i]-Y_analytique[i])
    erreur=erreur/n
    return erreur

'''fait la moyenne de la norme 2 d'erreur pour tout les points de la liste'''
def erreur_L2(Y_numerique,Y_analytique):
    erreur=0
    n=len(Y_numerique)
    for i in range(n):
        erreur+=(Y_numerique[i]-Y_analytique[i])**2
    erreur=erreur/n
    return np.sqrt(erreur)

'''fait la moyenne de la norme infini d'erreur pour tout les points de la liste'''
def erreur_Linf(Y_numerique,Y_analytique):
    erreur=[]
    n=len(Y_numerique)
    for i in range(n):
        erreur.append(abs(Y_numerique[i]-Y_analytique[i]))
    return max(erreur)

def erreur(Y_numerique,Y_analytique,norme):
    if norme==1:
        return erreur_L1(Y_numerique,Y_analytique)
    elif norme==2:
        return erreur_L2(Y_numerique,Y_analytique)
    else:
        return erreur_Linf(Y_numerique,Y_analytique)

'''affiche le graph de l'évolution de l'erreur en fonction du nombre de points de discrétisation pour l'exemple 1'''
def evolution_erreur_1(k,A,L,nb_points_max,Ta,Tb,norme):
    erreurs=[]
    for i in range(5,nb_points_max):
        X,Y_numerique,Y_analytique=list_graph_1(k,A,L,i,Ta,Tb)
        erreurs.append(erreur(Y_numerique,Y_analytique,norme))
    X=[i for i in range(5,nb_points_max)]
    plt.plot(X,erreurs)
    plt.ylim(0, max(erreurs)*1.1)
    plt.show()
    
'''affiche le graph de l'évolution de l'erreur en fonction du nombre de points de discrétisation pour l'exemple 2'''
def evolution_erreur_2(k,A,L,nb_points_max,Ta,Tb,q,norme):
    erreurs=[]
    for i in range(5,nb_points_max):
        X,Y_numerique,Y_analytique=list_graph_2(k,A,L,i,Ta,Tb,q)
        erreurs.append(erreur(Y_numerique,Y_analytique,norme))
    X=[i for i in range(5,nb_points_max)]
    plt.plot(X,erreurs)
    plt.ylim(0, max(erreurs)*1.1)
    plt.show()

'''affiche le graph de l'évolution de l'erreur en fonction du nombre de points de discrétisation pour l'exemple 3'''    
def evolution_erreur_3(k,A,L,nb_points_max,Ta,p,h,T_ambient,norme):
    erreurs=[]
    for i in range(5,nb_points_max):
        X,Y_numerique,Y_analytique=list_graph_3(k,A,L,i,Ta,p,h,T_ambient)
        erreurs.append(erreur(Y_numerique,Y_analytique,norme))
    X=[i for i in range(5,nb_points_max)]
    plt.plot(X,erreurs)
    plt.ylim(0, max(erreurs)*1.1)
    plt.show()

'''exemple 1'''
k,A,L,nb_points,Ta,Tb = 1000,5*10*10**(-3),0.1,50,100,500
nb_points_max=100
X,Y_numerique,Y_analytique=list_graph_1(k,A,L,nb_points,Ta,Tb)
graph(X,Y_numerique,Y_analytique)
evolution_erreur_1(k,A,L,nb_points_max,Ta,Tb,3)

'''exemple 2'''
k,A,L,nb_points,Ta,Tb,q = 0.5,1,0.004*5,50,100,200,1000*10**3
X,Y_numerique,Y_analytique=list_graph_2(k,A,L,nb_points,Ta,Tb,q)
graph(X,Y_numerique,Y_analytique)
evolution_erreur_2(k,A,L,nb_points_max,Ta,Tb,q,3)

'''exemple 3'''
k,A,L,nb_points,Ta,p,h,T_ambient = 1,1,0.2*5,50,100,1,25,20
X,Y_numerique,Y_analytique=list_graph_3(k,A,L,nb_points,Ta,p,h,T_ambient)
graph(X,Y_numerique,Y_analytique)
evolution_erreur_3(k,A,L,nb_points_max,Ta,p,h,T_ambient,3)