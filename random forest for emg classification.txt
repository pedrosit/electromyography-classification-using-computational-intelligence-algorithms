"""
Pedro Sito Monaco from the https://www.ufrgs.br/ieelab/
contact : pedro.smonaco@gmail.com

about the code:
    
Imports libraries needed for data analysis and machine learning.
Reads an Excel file containing feature and movement data.
Separates features and labels (target variables).
Splits the data into training and test sets.
Train a random forest classification model with the training data.
Use the trained model to predict the classes of the test data.
Evaluate the accuracy of the model using a confusion matrix and a classification report.
"""

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from warnings import filterwarnings

## read excel file generated in feature extraction

base = pd.read_excel('opa3.xlsx')

## chooses the values that will be used as input for the algorithm
## which in this case are the values of the characteristics extracted from the electromyography signal

carct = base.iloc[:,0:12].values

## chooses the values that will be used as model labels
## which in this case are the types of movements made by the volutary

previ = base.iloc[:,12].values

## Division of data
## Split one slice of the carcts into training and another into test and the same for the predictor
## 0.30 will divide 70 per center in training and 30 in testing of the carcts

from sklearn.model_selection import train_test_split

x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(
        carct,
        previ,
        test_size=0.30,
          
 )

## importing the random forest model

from sklearn.ensemble import RandomForestClassifier

algo = RandomForestClassifier()

## part where the model is trained

algo.fit(x_treinamento, y_treinamento)

## part that classifies the test data from the algorithm chosen in the line above
## where x_teste are the separate rms values in the test set

prevs = algo.predict(x_teste)

## confusion matrix as a tool for analyzing the assertiveness of the model

from sklearn.metrics import confusion_matrix

## pass y test which are the actuals and pass the predictions which are prevs with x test
## the y-axis of the checkered figure shows the actual classification of the data we had
## the x-axis shows the values that were classified by the algorithm
## by the amount of data classified right and wrong
## if the algorithm was good the largest values will be a diagonal

matr = confusion_matrix(y_teste, prevs)

## plotting the matrix to see the result as a graph

plt.figure(figsize=(10,5))

sns.heatmap(matr, annot=True)

## model assertiveness values in numerical reports 

from sklearn.metrics import classification_report

report = classification_report(y_teste, prevs)

print( report)

