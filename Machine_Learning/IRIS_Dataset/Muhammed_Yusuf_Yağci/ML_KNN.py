import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Veri yükleme

veriler = pd.read_csv("Iris.csv")
print(veriler)

# VERİ GÖRSELLEŞTİRME

df= veriler.iloc[:,2:6].copy()
sns.pairplot(data=veriler, hue="Species")
sns.pairplot(df)
plt.show()


# VERİ ÖN İŞLEME

boy = veriler[["SepalLengthCm"]]
#print(boy)


tür = veriler.iloc[:,-1].values
#print(tür) 

from sklearn import preprocessing

print(list(range(129)))

sonuc = pd.DataFrame(data=veriler, index= range(129), columns=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"])

#print(sonuc)

Species = veriler.iloc[:,-1].values

#print(Species)

sonuc1 = pd.DataFrame(data=veriler, index= range(129), columns=["Species"])

#print(sonuc1)

S = pd.concat([sonuc,sonuc1],axis=1)

print(S)

x = veriler.iloc[:,2:6].values
y = veriler.iloc[:,6:].values
#print(y)


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

print(x_train,x_test,y_train,y_test)

from sklearn.preprocessing import StandardScaler

sc= StandardScaler()

x_train = sc.fit_transform(x_train)
x_test  = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state=0)

logr.fit(x_train,y_train)


y_pred = logr.predict(x_test)

print(y_pred)
print(y_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
#print(cm)


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)






























