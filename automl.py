from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.cross_decomposition import CCA, PLSCanonical, PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import LinearSVR, SVR, NuSVR
from sklearn.neural_network import MLPRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Auto Regression Module
# This part of module create the training for a lot of different
# regressin algorithms from a list inside scikit-learn library
"""
"ARDRegression", "AdaBoostRegressor", "BaggingRegressor", "BayesianRidge",
"CCA", "DecisionTreeRegressor", "ElasticNet", "ElasticNetCV", "ExtraTreeRegressor",
"ExtraTreesRegressor", "GaussianProcessRegressor", "GradientBoostingRegressor", 
"HuberRegressor", "KNeighborsRegressor", "KernelRidge", "Lars", "LarsCV", "Lasso",
"LassoCV", "LassoLars", "LassoLarsCV", "LassoLarsIC", "LinearRegression",
"LinearSVR", "LogisticRegression", "LogisticRegressionCV", "MLPRegressor", 
"ModifiedHuber", "MultiTaskElasticNet", "MultiTaskElasticNetCV", "MultiTaskLasso",
"MultiTaskLassoCV", "NuSVR", "OrthogonalMatchingPursuit", "OrthogonalMatchingPursuitCV",
"PLSCanonical", "PLSRegression", "PassiveAggressiveRegressor", "RANSACRegressor",
"RadiusNeighborsRegressor", "RandomForestRegressor", "Ridge", "RidgeCV", "SGDRegressor",
"SVR", "TheilSenRegressor"
"""

# instantiate the algorithms 
# the names with brackets there is no opportunity to run with default template
# of instance

regressors = [
    linear_model.ARDRegression(),
    AdaBoostRegressor(),
    BaggingRegressor(),
    linear_model.BayesianRidge(),
    CCA(),
    DecisionTreeRegressor(),
    linear_model.ElasticNet(),
    linear_model.ElasticNetCV(),
    ExtraTreeRegressor(),
    ExtraTreesRegressor(),
    GaussianProcessRegressor(),
    GradientBoostingRegressor(random_state=50),
    linear_model.HuberRegressor(),
    KNeighborsRegressor(),
    KernelRidge(),
    linear_model.Lars(),
    linear_model.LarsCV(),
    linear_model.Lasso(),
    linear_model.LassoCV(),
    linear_model.LassoLars(),
    linear_model.LassoLarsCV(),
    linear_model.LassoLarsIC(),
    linear_model.LinearRegression(),
    LinearSVR(),
    #linear_model.LogisticRegression(),
    #linear_model.LogisticRegressionCV(),
    MLPRegressor(),
    #linear_model.ModifiedHuber(),
    #linear_model.MultiTaskElasticNet(),
    #linear_model.MultiTaskElasticNetCV(),
    #linear_model.MultiTaskLasso(),
    #linear_model.MultiTaskLassoCV(),
    NuSVR(),
    linear_model.OrthogonalMatchingPursuit(),
    linear_model.OrthogonalMatchingPursuitCV(),
    PLSCanonical(),
    PLSRegression(),
    linear_model.PassiveAggressiveRegressor(),
    linear_model.RANSACRegressor(),
    RadiusNeighborsRegressor(),
    RandomForestRegressor(),
    #linear_model.RandomizedLasso(),
    #linear_model.RandomizedLogisticRegression(),
    linear_model.RANSACRegressor(),
    linear_model.Ridge(),
    linear_model.RidgeCV(),
    linear_model.SGDRegressor(),
    SVR(),
    linear_model.TheilSenRegressor()]



# put the features in array format
array = df.values
X = array[:,0:14]
y = array[:, -1]


# cross validation dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
# standard format of scales to fitting inside the algorithms
X = StandardScaler().fit_transform(X)


class Regressors():
    """
    this class makes the auto training of default regression algorithms that is 
    inside scikit-learn library
    
    inputs:
    X_train
    y_train
    X_test
    y_test

    outputs:
    training: auto training for the list of algorithms
    score: generate a list of scores of each model
    score_sorted: generate a list of scores in sorted view
    models_sorted: generate a list of models in sorted view of scores
    predict: predict the values of each model
    confusion: generates the confusion matrix for each model

    """

    def __init__(self):
        pass
    
    def training(self, X_train, y_train):
        """fitting the traing set in the models"""
        for name, clf in zip(names, regressors):
            fit = clf.fit(X, y)
            yield fit
    
    def score (self, X_test, y_test):
        """get the names and scores of the models"""
        for name, results in zip(names, regressors):
            scored = results.score(X_test, y_test)
            yield name, scored
            
    def score_sorted(self, X, y):
        """"return the score of each models"""
        for results in regressors:
            score = results.score(X, y)
            yield score
    
    def models_sorted(self, names, beta):
        """return the models with name in sorted position by accuracy"""
        modelos = [val for pair in zip(names, sorted(beta, reverse=True)) for val in pair]
        it = iter(modelos)
        for x in it:
            print ("Modelo {} tem score igual a {:.3f}".format(x, next(it)))
            
    def predict(self, names, regressors, X_train, y_train, X_test):
        """return the predict of the all models trained"""
        for name, prediction in zip(names, regressors):
            y_pred = prediction.fit(X_train, y_train).predict(X_test)
            yield name, y_pred
    
    def confusion(self, y_test):
        """"return the confusion matrix for models"""
        for conf in a:
            cnf_matrix = confusion_matrix(y_test, conf)
            yield cnf_matrix
