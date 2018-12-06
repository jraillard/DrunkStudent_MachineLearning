import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import preprocessing,metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

###################
# DATASET
###################
data_df = pd.read_csv("../student-mat.csv")


###################
# DATA PREPARATION
###################
data_df = data_df.drop('Id',axis=1)
data_df = data_df.replace(
    ['F','M','U', 'R','LE3','GT3','no','yes','A','T','teacher','health','services','at_home','other','mother','father'],
    [0,1,0,1,0,1,0,1,0,1,1,2,3,4,0,1,2])


##################
# Chosen Descriptors
###################
# data_df=data_df[['age','traveltime','failures','freetime', 'goout','health', 'absences','Dalc','Walc']]

##################
# All Descriptors
###################
data_df=data_df[['sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
       'Mjob', 'Fjob', 'guardian', 'traveltime', 'studytime',
       'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
       'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
       'Walc', 'health', 'absences', 'G1', 'G2', 'G3']]

##################
# SEPARATION INPUTS (VARIABLES)/OUTPUTS (LABELS-TARGET)
###################
train_data_Walc, test_data_Walc = train_test_split(data_df, test_size=0.2, random_state=42)
train_data_Dalc, test_data_Dalc = train_test_split(data_df, test_size=0.2, random_state=42)

# Train
train_data_labels_Dalc = train_data_Dalc["Dalc"].copy()
train_data_Dalc = train_data_Dalc.drop("Dalc", axis=1)

train_data_labels_Walc = train_data_Walc["Walc"].copy()
train_data_Walc = train_data_Walc.drop("Walc", axis=1)

# Test
test_data_labels_Dalc = test_data_Dalc["Dalc"].copy()
test_data_Dalc = test_data_Dalc.drop("Dalc", axis=1)

test_data_labels_Walc = test_data_Walc["Walc"].copy()
test_data_Walc = test_data_Walc.drop("Walc", axis=1)

##################
# CLASSIFICATION
##################
for i in [1,2,3,5,7,10,15,50]:
    Tree_Dalc=DecisionTreeClassifier(max_depth=i)
    Tree_Dalc.fit(train_data_Dalc,train_data_labels_Dalc)
    predicted_labels_Dalc=Tree_Dalc.predict(test_data_Dalc) 

    Tree_Walc=DecisionTreeClassifier(max_depth=i)
    Tree_Walc.fit(train_data_Walc,train_data_labels_Walc)
    predicted_labels_Walc=Tree_Walc.predict(test_data_Walc) 

    ##################
    # PREDICTION ERROR
    ##################
    print("\n",i)
    acc=sk.metrics.accuracy_score(test_data_labels_Dalc, predicted_labels_Dalc)
    print("Taux de mauvaise classification Dalc:\n",1-acc)

    acc=sk.metrics.accuracy_score(test_data_labels_Walc, predicted_labels_Walc)
    print("Taux de mauvaise classification Walc:\n",1-acc)