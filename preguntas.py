"""
Detección de hongos venenosos usando Regresión Logistica
-----------------------------------------------------------------------------------------

Construya un modelo de regresión logística que permita identificar si un hongo es 
venenoso o no. Para ello, utilice la muestra de datos suministrada. 

La base de datos contiene 8124 instancias de hongos provenientes de 23 especies de la 
familia Agaricus y Lepiota, los cuales han sido clasificados como comestibles, venenosos
o de comestibilidad indeterminada. Por el tipo de problema en cuestión, los hongos de 
comestibilidad desconocida deben ser asignados a la clase de hongos venenosos, ya que no
se puede correr el riesgo de dar un hongo potencialmente venenoso a una persona para su 
consumo.

Véase https://www.kaggle.com/uciml/mushroom-classification

Evalue el modelo usando la matriz de confusión.

La información contenida en la muestra es la siguiente:

     1. cap-shape:                bell=b,conical=c,convex=x,flat=f,
                                  knobbed=k,sunken=s
     2. cap-surface:              fibrous=f,grooves=g,scaly=y,smooth=s
     3. cap-color:                brown=n,buff=b,cinnamon=c,gray=g,green=r,
                                  pink=p,purple=u,red=e,white=w,yellow=y
     4. bruises?:                 bruises=t,no=f
     5. odor:                     almond=a,anise=l,creosote=c,fishy=y,foul=f,
                                  musty=m,none=n,pungent=p,spicy=s
     6. gill-attachment:          attached=a,descending=d,free=f,notched=n
     7. gill-spacing:             close=c,crowded=w,distant=d
     8. gill-size:                broad=b,narrow=n
     9. gill-color:               black=k,brown=n,buff=b,chocolate=h,gray=g,
                                  green=r,orange=o,pink=p,purple=u,red=e,
                                  white=w,yellow=y
    10. stalk-shape:              enlarging=e,tapering=t
    11. stalk-root:               bulbous=b,club=c,cup=u,equal=e,
                                  rhizomorphs=z,rooted=r,missing=?
    12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
    13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
    14. stalk-color-above-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,
                                  pink=p,red=e,white=w,yellow=y
    15. stalk-color-below-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,
                                  pink=p,red=e,white=w,yellow=y
    16. veil-type:                partial=p,universal=u
    17. veil-color:               brown=n,orange=o,white=w,yellow=y
    18. ring-number:              none=n,one=o,two=t
    19. ring-type:                cobwebby=c,evanescent=e,flaring=f,large=l,
                                  none=n,pendant=p,sheathing=s,zone=z
    20. spore-print-color:        black=k,brown=n,buff=b,chocolate=h,green=r,
                                  orange=o,purple=u,white=w,yellow=y
    21. population:               abundant=a,clustered=c,numerous=n,
                                  scattered=s,several=v,solitary=y
    22. habitat:                  grasses=g,leaves=l,meadows=m,paths=p,
                                  urban=u,waste=w,woods=d


"""

import pandas as pd


def pregunta_01():
    df = pd.read_csv('mushrooms.csv',sep=",")
    
    df.drop('veil_type',axis=1,inplace=True)
    
    y = df['type']
    
    X = df.copy(deep=True)
    
    X.drop('type',axis=1,inplace=True)
    
    return X, y


def pregunta_02():
    from sklearn.model_selection import train_test_split
    
    X, y = pregunta_01()
    
    (X_train, X_test, y_train, y_test,) = train_test_split(
        X,
        y,
        test_size=50,
        random_state=123,
    )
    
    return X_train, X_test, y_train, y_test


def pregunta_03():
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    
    X_train, X_test, y_train, y_test = pregunta_02()
    
    pipeline = Pipeline(
        steps=[
            ("onehot", OneHotEncoder()),
            ("logisticRegression", LogisticRegressionCV(Cs=10)),
        ],
    )
    
    pipeline.fit(X_train, y_train)
    
    return pipeline


def pregunta_04():
    from sklearn.metrics import confusion_matrix
    
    pipeline = pregunta_03()
    
    X_train, X_test, y_train, y_test = pregunta_02()
    
    cfm_train = confusion_matrix(
        y_true=y_train,
        y_pred=pipeline.predict(X_train),
    )

    cfm_test = confusion_matrix(
        y_true=y_test,
        y_pred=pipeline.predict(X_test),
    )
    
    return cfm_train, cfm_test
