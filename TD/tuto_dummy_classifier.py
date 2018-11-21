import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import preprocessing,metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import math

##################
# DUMMY CLASSIFIER
##################
class DummyClassifier(sk.base.BaseEstimator):
    def __init__(self,threshold=None): self.threshold=threshold
    def fit(self,data,labels):
        if self.threshold is None: self.threshold=np.median(data.values[:,0])
    def predict(self,data): #pour chaque valeur < threshold, on affecte la classe 1, sinon 0
        return np.where(data.values.flatten()<self.threshold,1,0)

def proba(data , theta, sigma):
    return (1 / (math.sqrt(sigma*2*np.pi))) * math.exp(-(data-theta)**2/(2*sigma))

############################
# Chargement CSV: 50 first lines (2 classes) et un seul descripteur (SepalWidthCm)
data_df = pd.read_csv('Iris.csv')
data_df=data_df.iloc[:100,:]
data_df=data_df[['SepalWidthCm','Species']] #;print(data_df.head(-1));quit()

############################
# Encode class names: 'Iris-setosa' -> 0 ; 'Iris-versicolor' -> 1
le = preprocessing.LabelEncoder()
le.fit(['Iris-setosa','Iris-versicolor'])
data_df['Species']=le.transform(data_df['Species']) #print(data_df.head(-1));quit()

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
# clf=DummyClassifier(threshold =3.3) #threshold can be set manually: clf=DummyClassifier(threshold=???)
# clf.fit(train_data,train_data_labels) #training
# predicted_labels=clf.predict(test_data) #predicting

# neighbors=sk.neighbors.KNeighborsClassifier(n_neighbors=7)
# neighbors.fit(train_data,train_data_labels)
# predicted_labels=neighbors.predict(test_data) #predicting

# Bayes=GaussianNB()
# Bayes.fit(train_data,train_data_labels)
# predicted_labels=Bayes.predict(test_data) #predicting
# print("theta[setosa,versicolor] = ",Bayes.theta_,"\n")
# print("sigma[setosa,versicolor] = ",Bayes.sigma_,"\n")
# print("p(setosa) = ", Bayes.class_prior_)
# print("p(3|setosa) = ", (proba(3,3.39,0.15)*0.475)/(proba(3,3.39,0.15)+proba(3,2.78,0.01)))
# print("p(3|versicolor) = ", (proba(3,2.78,0.01)*0.525)/(proba(3,3.39,0.15)+proba(3,2.78,0.01)))



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


