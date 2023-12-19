# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 13:12:58 2022

@author: Group "rainbow expert junglefowl"
         Carlos de Cloet
         Roel Veth
         Raslen Kouzana
         Yme Bartels
         
         Extreme Gradient Boosting Model to predict poverty in Costa Rican Households.

"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix


rnd_nm = 3

df_trainingset = pd.read_csv('train.csv',';', decimal=',')

X = df_trainingset.drop('target', axis = 1).copy()
y = df_trainingset['target']

prediction_set = pd.read_csv('test.csv',';', decimal=',')


percentage_poor = sum(y)/len(y)

# Splits data in training and test data.

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = rnd_nm, stratify = y)

###### Tuning of the hyper-parameters ###############
#
# Round 1
#
# param_grid = {'max_depth': [ 4, 5, 6],
#               'learning_rate': [0.15, 0.2, 0.25],
#               'gamma': [0.1, 0.2, 0.3, 0.4],
#               'reg_lambda': [0.0, 0.1, 0.2],
#               'scale_pos_weight': [1, 1.3, 1.65]}
#
# Round 2
#
# #param_grid = {'max_depth': [ 4, 5, 6],
#               'learning_rate': [0..15, 0.2, 0.25],
#               'gamma': [0.1, 0.2, 0.3, 0.4],
#               'reg_lambda': [0.0, 0.1, 0.2],
#               'scale_pos_weight': [1.65]}
#
# Round 3
#
# param_grid = {'max_depth': [6],
#               'learning_rate': [ 0.2, 0.3, 0.4],
#               'gamma': [ 0.05, 0.1, 0.15],
#               'reg_lambda': [0.0, 0.1, 0.2],
#               'scale_pos_weight': [1.65]}
#
# Round 4
#
# param_grid = {'max_depth': [ 4, 5, 6],
#               'learning_rate': [0.15, 0.2, 0.25],
#               'gamma': [0.1, 0.2, 0.3, 0.4],
#               'reg_lambda': [0.0, 0.1, 0.2],
#               'scale_pos_weight': [1, 1.3, 1.65]}
#
#
# optimal_params = GridSearchCV(
#    estimator = xgb.XGBClassifier(objective = 'binary:logistic',
#                                  method = 'hist',
#                                  seed = rnd_nm+10,
#                                  n_estimators = 300,
#                                  subsample = 0.9,
#                                  colsample_bytree = 0.5),
#                 param_grid = param_grid,
#                 scoring = 'accuracy',
#                 verbose = 0,
#                 n_jobs = -1,
#                 cv = 5 )
#
# optimal_params.fit(X_train,
#                    y_train,
#                    early_stopping_rounds = 50,
#                    eval_metric = 'error',
#                    eval_set = [(X_test, y_test)],
#                    verbose = False)
#
# print(optimal_params.best_params_)
##################################################

# Fitting and evaluation
opt_xgb = xgb.XGBClassifier(objective = 'binary:logistic',
                                  method = 'hist',
                                  seed = rnd_nm+20,
                                  n_estimators = 300,
                                  gamma =0.1,
                                  learning_rate =0.25,
                                  max_depth=6,
                                  reg_lambda=0.0,
                                  scale_pos_weight=1.65,
                                  )
    
opt_xgb.fit(X_train,
            y_train,
            verbose=True,
            early_stopping_rounds = 300,
            eval_metric = 'error',
            eval_set = [(X_test, y_test)])


yhat_train = opt_xgb.predict(X_train)
yhat_test = opt_xgb.predict(X_test)
print()
print('The in-sample accuracy is: ' + str(accuracy_score(y_train, yhat_train)))
print('The out-of-sample accuracy is: ' + str(accuracy_score(y_test, yhat_test)))
print()

plot_confusion_matrix(opt_xgb,
                      X_test,
                      y_test)

# Final model for prediction

fin_xgb = xgb.XGBClassifier(objective = 'binary:logistic',
                                  method = 'hist',
                                  seed = rnd_nm+20,
                                  n_estimators = 300,
                                  gamma =0.1,
                                  learning_rate =0.25,
                                  max_depth=6,
                                  reg_lambda=0.0,
                                  scale_pos_weight=1.65,
                                  )
    
fin_xgb.fit(X,
            y,
            verbose=True)

predictions = fin_xgb.predict(prediction_set)


# Writes the predictions to a txt file

with open('predictions.txt', 'w') as f:
    
    for i in range(len(predictions)):
        f.write(str(predictions[i]))
        
        