________________________________________________________
#Iris


#Ce projet constitue un regroupement de méthodes différentes
#en machine learning appliqués au dataset "Iris". On essayera
#de prédire la classification des fleurs (et toute autre
#variable)


________________________________________________________
#Fonction reset


#Pour effacer les variables
def reset():
    df=pd.read_csv("D:\Anaconda\Projets\Iris Flower\iris.csv")
    global X,y,X_train,X_test,y_train,y_test
    del X,y,X_train,X_test,y_train,y_test
    print("Les variables ont été effacées")
reset()


________________________________________________________
#Régression linéaire


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
%matplotlib inline
reset()

df=pd.read_csv("D:\Anaconda\Projets\Iris Flower\iris.csv")
df
sns.set_style("whitegrid")

#Regardons la vue d'ensemble des résultats
sns.pairplot(df,hue="species")
plt.matshow(df.corr())
#Nous pouvons observer qu'il y a une forte corrélation entre
#petal width et petal length

#Faisons une régression linéaire, avec petal_width étant la 
#variable à observer

from sklearn.model_selection import train_test_split

y=df["petal_width"]
X=df[["sepal_length","sepal_width","petal_length"]]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)

#On va séparer nos donnés en 2 parties: un training set et 
#un testing set

from sklearn.linear_model import LinearRegression

lm=LinearRegression()
lm.fit(X_train,y_train)

#Ceci sera notre estimateur. Avec les données du training set
#de X et de y, il déterminera une droite

print("Les coefficients sont:", lm.coef_)

#Maintenant, testons notre modèle

pred=lm.predict(X_test)
plt.scatter(y_test,pred)

#Est-ce que notre modèle est bon?
sns.distplot((y_test-pred),bins=50)
#Le modèle ressemble à une courbe normale. Il n'y a pas 
#beaucoup de overfit

coeficients=pd.DataFrame(lm.coef_,X.columns)
coeficients.columns=["coeficients"]
coeficients

#Donc, si la longueur du pétale monte de 1cm, la largeur 
#de celle-ci augmentera de 0.550335cm. 

#On peut voir que les sépales ne sont pas fortement corrélés 
#avec la grandeur des pétales (ce qui est logique)







________________________________________________________
#Random Forest


#Nous allons prédire si une fleur appartient à l'espèce de
#sétosa ou non. Regardons les données

reset()

df[df["species"]=="setosa"]["petal_width"].hist(bins=50, label="Setosa")
df[df["species"]!="setosa"]["petal_width"].hist(bins=50, label="Pas setosa")

df[df["species"]=="setosa"]["sepal_width"].hist(bins=50, label="Setosa")
df[df["species"]!="setosa"]["sepal_width"].hist(bins=50, label="Pas setosa")

#Changeons les données pour avoir des données catégoriques
for i in range(1,len(df["species"])):
    if df["species"][i]!="setosa":
        df["species"][i]="pas_setosa"
df
final_data=pd.get_dummies(df,"species",drop_first=True)
final_data

#Split les données
from sklearn.model_selection import train_test_split
X=final_data.drop("species_setosa",axis=1)
y=final_data["species_setosa"].astype(int)
y
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30)

#Entrainer les données
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=700)
rfc.fit(X_train,y_train)

#Prédiction et conclusion
from sklearn.metrics import confusion_matrix, classification_report
predictions=rfc.predict(X_test)
print("Voici la matrice de confusion et le rapport de classification:")
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))







________________________________________________________
#SVM - Support Vector Machine (à partir de Udemy)


reset()

#En regardant le pairplot de la régression linéaire, on voit
#que la setosa est différente des autres

#Split les données
from sklearn.cross_validation import train_test_split
X=df.drop("species",axis=1)
y=df["species"]
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.4)

#Grid Search (Prendre le meilleur modèle)
from sklearn.model_selection import GridSearchCV
parametres={"C":[0.1,1,10,100],"gamma":[1,0.1,0.01,0.001]}
g=GridSearchCV(SVC(),parametres,refit=True,verbose=2)
g.fit(X_train,y_train)

#Prédiction et conclusion
from sklearn.metrics import confusion_matrix, classification_report
predictions=g.predict(X_test)
print("Voici la matrice de confusion et le rapport de classification:")
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))







________________________________________________________
#k-NN - K-Nearest Neighbors


reset()

#Split les données
X=df.drop("species",axis=1)
y=df["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Entrainer les données
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

#Prédiction et conclusion
pred = knn.predict(X_test)
print("Voici la matrice de confusion et le rapport de classification:")
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
