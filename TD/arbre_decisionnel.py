import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import preprocessing,metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier,RandomForestClassifier
import math



############################
# Chargement CSV: 50 first lines (2 classes) et un seul descripteur (SepalWidthCm)
data_df = pd.read_csv("Iris.csv")
data_df=data_df.iloc[:150,:]
data_df=data_df[['SepalLengthCm','SepalWidthCm','Species']] #;print(data_df.head(-1));quit()

############################
# Encode class names: 'Iris-setosa' -> 0 ; 'Iris-versicolor' -> 1
le = preprocessing.LabelEncoder()
le.fit(['Iris-setosa','Iris-versicolor','Iris-virginica'])
data_df['Species']=le.transform(data_df['Species']) # print(data_df.head(-1));quit()

##################
# SEPARATION INPUTS (VARIABLES)/OUTPUTS (LABELS-TARGET)
train_data, test_data = train_test_split(data_df, test_size=0.2, random_state=42)
# Train
train_data_labels = train_data["Species"].copy()
train_data = train_data.drop("Species", axis=1)
# Test
test_data_labels = test_data["Species"].copy()
test_data = test_data.drop("Species", axis=1)

##################
# CLASSIFICATION

# clf1 = sk.neighbors.KNeighborsClassifier(n_neighbors=5)
# clf2 =DecisionTreeClassifier(max_depth=2)

# clf = VotingClassifier(estimators=[('kn', clf1),('dt', clf2)],voting='soft')
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(train_data,train_data_labels)
predicted_labels=clf.predict(test_data) #predicting
# print(clf.predict(test_data))


##################
# EVALUATION: QUELQUES METRIQUES
confusion=sk.metrics.confusion_matrix(test_data_labels, predicted_labels)
print("Matrice de confusion [setosa,versicolor] versus [setosa,versicolor]:\n",confusion)
acc=sk.metrics.accuracy_score(test_data_labels, predicted_labels)
print("Taux de bonne classification:\n",acc)
precision=sk.metrics.precision_score(test_data_labels, predicted_labels,average=None)
print("Precision par rapport a [setosa,versicolor]:\n",precision)
rappel=sk.metrics.recall_score(test_data_labels, predicted_labels,average=None)
print("Rappel par rapport a [setosa,versicolor]:\n",rappel)

##################
# EVALUATION: EXEMPLE DE METRIQUE BINAIRE
vn,fn,fp,vp=confusion[0,0],confusion[0,1],confusion[1,0],confusion[1,1]
print("Versicolor: vn=",vn,", fn=",fn,", fp=",fp,", vp=",vp)


