
#%%
from meanSquare import MeanSquare

#%%
ms = MeanSquare()


#%%
param = {0: ('L', 0.0),1:('L',0.0),2:('L',0,0),3:('L',0.0)}


#%%
ms.createCL(param)


#%%
def fct(x,y):
    return x**2 + y**2

def grad(x,y):
    return [2*x,2*y]

#%%
ms.setChamp(fct,grad)

#%%
ms.createElements()

#%%
ms.constructionATA()

#%%
ms.constructionB()

#%%
ms.calculMeanSquare()
ms.debug()