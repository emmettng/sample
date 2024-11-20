import logging
from a_preprocess import *

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, precision_score, recall_score,accuracy_score,make_scorer,confusion_matrix,precision_recall_curve
from sklearn.externals import joblib

logging.basicConfig(filename = './gridsearch_'+'.log',format = '%(asctime)s %(message)s',level = logging.DEBUG)
MODELNAME = "clf.pkl"
GRIDMODEL = "grid_clf.pkl"

def _naiveGridSearchCV(X,y,clf,param_grid,scores,refit_metric):
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2,stratify=y)
    print ("Start grid search training")
    skf = StratifiedKFold(n_splits=5)
    grid_clf= GridSearchCV(clf,param_grid,scoring=scores,refit=refit_metric,cv=skf,return_train_score=True)
    grid_clf.fit(X_train,y_train)

    modelPath = getModelPath()

    for k, v in param_grid.items():
        logging.info("Best %s is: %s" % (k,grid_clf.best_estimator_.get_params()[k]))
    y_predict = grid_clf.predict(X_test)
    y_proba = grid_clf.predict_proba(X_test)
    adThreshold_1 = _adjustThreshold(y_test,y_proba,modelPath+"training",1)
    logging.info("For label 1, recall score > precision score, when threshold biger than %f" % adThreshold_1)
    adThreshold_0 = _adjustThreshold(y_test,y_proba,modelPath+"training",0)
    logging.info("For label 0, recall score > precision score, when threshold biger than %f" % adThreshold_0)
    acu = accuracy_score(y_test, y_predict)
    report = classification_report(y_test,y_predict)
    logging.info("acu on test set: %f" % acu)
    logging.info("Classification report: %s" % (report))
    savepath = modelPath+GRIDMODEL
    print ("Saving model to %s" % savepath)
    joblib.dump(grid_clf,savepath)

def naiveTraining(v_tag,scaler,clf,hyperGrid,refit_metric):
    v_tag = v_tag+"_"+scaler+"_"+refit_metric
    X,y = naivePreprocess(v_tag=v_tag)          ## basci process
    X = featureScaling(X,scaler)              ## scaling the input data
    modelPath = getModelPath()

    scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score)
    }
    X_train, X_valid, y_train, y_valid= train_test_split(X, y.values, test_size=0.2,stratify=y)
    _naiveGridSearchCV(X_train,y_train,clf,hyperGrid,scorers,refit_metric)

    Save2pkl(X_valid,modelPath+"X_valid.pkl")
    Save2pkl(y_valid,modelPath+"y_valid.pkl")

    return (X_valid, y_valid,modelPath)

def naiveValidation(X_valid,y_valid,modelPath,relevant=1,newThreshold=None):
    grid_clf = joblib.load(modelPath+GRIDMODEL)

    y_predict = grid_clf.predict(X_valid)
    acu = accuracy_score(y_valid, y_predict)
    p = precision_score(y_valid,y_predict)
    r = recall_score(y_valid,y_predict)
    report = classification_report(y_valid, y_predict)
    logging.info("Validation information below:")
    logging.info("acu on test set: %f" % acu)
    logging.info("precision on valid set: %f" % p)
    logging.info("recal on valid set: %f" % r)
    logging.info("Classification report: %s" % (report))

    y_proba = grid_clf.predict_proba(X_valid)

    cm = confusion_matrix(y_valid,y_predict)
    logging.info(str(cm))

    _ = _adjustThreshold(y_valid,y_proba,modelPath+"validation")


def _adjustThreshold(y_true,y_proba,imgPath,relevant=1):
    y_scores = y_proba[:,relevant]
    p, r, thresholds = precision_recall_curve(y_true, y_scores)
    plt.figure(figsize=(8, 8))
    plt.title("(Precision, Recall) -> Decision Threshold")
    plt.plot(thresholds, p[:-1], "b--", label="Precision")
    plt.plot(thresholds, r[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')
    seg = list()
    index = 0
    for i,v in enumerate((map(lambda x: x[0]<x[1],zip(p,r)))):
        if v:
            index = i
            continue
        seg.append(v)
    if not any(seg):
        crossPoint = thresholds[index]
    imgPath = imgPath+str(relevant)+".png"
    plt.savefig(imgPath)
    return crossPoint


def _trainingParams(task,scaler,gridDict):
    trainingDict = dict()
    trainingDict[task] = (gridDict,scaler)
    return trainingDict

def trainingAndValidation():
    refit_metrics = ["recall_score"]

    '''
    1.Logistic regression
    '''
    clf = LogisticRegression(random_state=666, max_iter=1000, class_weight='balanced')
    featureDict = _trainingParams(task='training_test_lr',
                                 scaler='MinMax',
                                 gridDict = dict(
                                        penalty=['l1'],
                                        C=[0.01],
                                        solver=['liblinear']
                                        )
                                  )

    '''
    2.Random forest
    '''
    # clf = RandomForestClassifier(class_weight='balanced')
    # featureDict = _trainingParams(
    #                             task="training_test_rf",
    #                             scaler = 'MinMax',
    #                             gridDict = dict(
    #                                         n_estimators = [10,],
    #                                         max_depth=[5,],
    #                                         max_features=[8,],
    #                                         criterion=["gini"]
    #                             )
    # )

    '''
    Grid Search training
    '''
    trainingDict=featureDict

    for tag, v in trainingDict.items():
        hypers, scaler = v
        for refit_metric in refit_metrics:
            X_valid, y_valid, modelpath= naiveTraining(tag,scaler,clf,hypers,refit_metric)
            naiveValidation(X_valid,y_valid,modelpath)

def thresholdMetrics():
    return True

if __name__== "__main__":
    '''
    1. Training process and validation process.
    '''
    trainingAndValidation()

    '''
    2. Demonstration of validation process.
    '''
    # model_key = _
    # X_valid = joblib.load(getModelPath()+model_key+"X.pkl")
    # y_valid = joblib.load(getModelPath()+model_key+"y.pkl")
    # naiveValidation(X_valid,y_valid,getModelPath()+model_key,1,0.45)

