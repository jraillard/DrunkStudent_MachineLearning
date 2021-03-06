import numpy as np
import pandas as pd
import sklearn as sk
from matplotlib import pyplot as plt
from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error

###################
# DATASET
###################
data = pd.read_csv("../student-mat.csv")


###################
# DATA PREPARATION
###################

data = data.drop('Id',axis=1)
data = data.drop('school',axis=1)
data = data.drop('reason', axis=1)
data = data.replace(
    ['F','M','U', 'R','LE3','GT3','no','yes','A','T','teacher','health','services','at_home','other','mother','father'],
    [0,1,0,1,0,1,0,1,0,1,1,2,3,4,0,1,2])

##################
# Chosen Descriptors
###################
# data=data[['age','traveltime','failures','freetime', 'goout','health', 'absences','Dalc','Walc']]

##################
# All Descriptors
###################
data=data[['sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
       'Mjob', 'Fjob', 'guardian', 'traveltime', 'studytime',
       'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
       'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
       'Walc', 'health', 'absences', 'G1', 'G2', 'G3']]

###################
# PREPARATION TRAINING/TESTING
###################
training=data.iloc[:316,:]
testing=data.iloc[79:,:]

###################
# PREPARATION INPUT/OUTPUT
###################
training_labels_Dalc = training['Dalc'].copy()
training_Dalc  = training.drop('Dalc', axis=1)
testing_labels_Dalc  = testing['Dalc'].copy()
testing_Dalc  = testing.drop('Dalc', axis=1)

training_labels_Walc  = training['Walc'].copy()
training_Walc = training.drop('Walc', axis=1)
testing_labels_Walc = testing['Walc'].copy()
testing_Walc = testing.drop('Walc', axis=1)

###################
# LEARNING: Y=AX_1+BX_2+C
###################
model_Dalc = linear_model.LinearRegression()
model_Dalc.fit(training_Dalc, training_labels_Dalc)

model_Walc = linear_model.LinearRegression()
model_Walc.fit(training_Walc, training_labels_Walc)

###################
#  PREDICTION: UNKNOWN INPUT -> OUTPUT
###################
predicted_labels_Walc=model_Walc.predict(testing_Walc)
predicted_labels_Dalc=model_Dalc.predict(testing_Dalc)

###################
#  EVALUATION
###################
lin_mse = mean_squared_error(testing_labels_Dalc, predicted_labels_Dalc)
lin_rmse = np.sqrt(lin_mse)
print("Error Dalc (testing): ",(lin_rmse/5)*100)

lin_mse = mean_squared_error(testing_labels_Walc, predicted_labels_Walc)
lin_rmse = np.sqrt(lin_mse)
print("Error Walc (testing): ",(lin_rmse/5)*100)