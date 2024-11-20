from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import os
import datetime
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pickle
import itertools
import random

def _h_trans_predict(Ys):
    y_predict = list()
    for y in Ys:
        if type(y) is float:
            y_predict.append(round(y))
        else:
            tmpMax = np.max(y)
            y_predict.append(list(y).index(tmpMax))
    return y_predict

def _get_params_generator(early_stop=False):

    commonParamListDict = {
        "objective": "multiclass",
        "boosting": ["gbdt"],
        # "boosting": ["dart"],
        #        "num_class": 3,
        #        "num_class": 2,
        # "num_leaves": [100,30],
        # "num_leaves": [55],
        "max_depth": -1,
        "learning_rate": [0.2],
        # "bagging_fraction": [0.3],  # subsample
        # "feature_fraction": [0.9],  # colsample_bytree
        # "bagging_freq": 5,  # subsample_freq
        # "max_bin":[50,255,500],
        "bagging_seed": 2019,
        "verbosity": -1,
        # "lambda_l2": 1,
        "lambda_l1": 1,
        "is_unbalance":True,
        # "scale_pos_weight": [1,0.25],
        # "scale_pos_weight": [0.5],
        "min_sum_hessian_in_leaf": 0.002,
        "num_iterations": [3000],
    }

    earlyStopParamListDict= {
        "objective": "multiclass",
        "boosting": ["gbdt"],
        "num_class": 18,
        "bagging_fraction": [0.3],  # subsample
        "feature_fraction": [0.5],  # colsample_bytree
        "bagging_freq": 2,  # subsample_freq
        "bagging_seed": 2019,
        "verbosity": -1,
        "lambda_l2": 2,
        # "lambda_l1": 5,
        "is_unbalance":True,
        "min_sum_hessian_in_leaf": 0.002,
        # 'objective': 'binary',
        'metric': ['multi_logloss'],
        # 'metric': "None",
        'learning_rate': 0.002,
        'num_leaves': 15,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        # 'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 10,  # Number of bucketed bin for feature values
        # "scale_pos_weight": 8000,
        'nthread': 4,
    }

    if not early_stop:
        paramListDict = commonParamListDict
    else:
        paramListDict = earlyStopParamListDict


    tmpParaList = list()
    tmpValueList = list()
    paraDict = dict()
    for k,v in paramListDict.items():
        if type(v) is list:
            tmpParaList.append(k)
            tmpValueList.append(v)
            continue
        paraDict[k] = v

    iterValueList = list(itertools.product(*tmpValueList))

    paraDictList = list()

    for value in iterValueList:
        nDict = dict(zip(tmpParaList,value))
        newDict = {**paraDict,**nDict}
        paraDictList.append(newDict)

    for pd in paraDictList:
        yield  pd


def _singleEarlyStopping(trainData,validData,orgSet,cusParam):
    EARLY_STOP = True
    evals_results = {}
    params_generator = _get_params_generator(EARLY_STOP)
    x_train,y_train,x_valid,y_valid = orgSet
    # _wh_eval={}
    if cusParam is None:
        for params in params_generator:
            print(params)
            lgbm = lgb.train(params,
                             trainData,
                             valid_sets=validData,
                             evals_result=evals_results,
                             num_boost_round=20000,
                             early_stopping_rounds=4000,
                             verbose_eval=500,
                             # feval = _wh_eval
                             )


    else:
        lgbm = lgb.train(
            cusParam,
            trainData,
            valid_sets=validData,
            evals_result=evals_results,
            num_boost_round=200000,
            early_stopping_rounds=10000,
            verbose_eval=500,
        )

    ypred_valid= lgbm.predict(x_valid,num_iteration=lgbm.best_iteration)
    y_predict_valid = _h_trans_predict(ypred_valid)
    report_valid = classification_report(y_valid,y_predict_valid)
    conMatrix_valid= confusion_matrix(y_valid,y_predict_valid)
    print (report_valid)
    print (conMatrix_valid)

    return lgbm

def singleTraining(trainList,validList,params=None):
    print ("")
    print ("start the training stage !")
    nx,ny  = trainList
    nvx,nvy= validList

    trainData = lgb.Dataset(nx,ny)
    validData = lgb.Dataset(nvx,nvy)

    lgbm = _singleEarlyStopping(trainData,validData,[nx,ny,nvx,nvy],params)

    return lgbm

def _dGenerator(ds,num):
    ds1,ds2 = ds
    dds1 = np.split(np.asarray(ds1[:(len(ds1) - len(ds1)%num)]),num)
    dds2 = np.split(np.asarray(ds2[:(len(ds2) - len(ds2)%num)]),num)
    ddds = list()
    for i,d in enumerate(dds1):
        ddds.append([d,dds2[i]])
    return ddds


def _transStack(stackXs):
    l = len(stackXs[0])
    n = len(stackXs)
    trans=list()
    for il in range(l):
        tmpL = []
        for ii in range(n):
            tmpL = tmpL+list(stackXs[ii][il])
        trans.append(tmpL)
    print (l)
    print (n)
    print (len(trans))
    print (len(trans[0]))
    print (trans[0])
    return trans

def stackingLGBM(skuPair,serializer,stackSize,upperTrain,upperTest):
    stackTrain = list()
    stackTest = list()
    for i in range(stackSize):
        modelName = "_".join(list(skuPair)) + "_" + str(i) + ".txt"
        bst, _ = serializer.loadBooster(modelName)
        x, _ = upperTrain
        vx, _ = upperTest
        tmpX = bst.predict(x)
        stackTrain.append(tmpX)
        tmpVX = bst.predict(vx)
        stackTest.append(tmpVX)

    stackTrainFeature = _transStack(stackTrain)
    stackTestFeature = _transStack(stackTest)


    _, y = upperTrain
    _, vy = upperTest

    nx = np.asarray(stackTrainFeature)
    ny = np.asarray(y)

    nvx = np.asarray(stackTestFeature)
    nvy = np.asarray(vy)
    cusParams = {
        "objective": "multiclass",
        "boosting": ["gbdt"],
        "num_class": 18,
        "bagging_freq": 2,  # subsample_freq
        "bagging_seed": 2019,
        "verbosity": -1,
        "lambda_l2": 2,
        # "lambda_l1": 5,
        "is_unbalance": True,
        "min_sum_hessian_in_leaf": 0.002,
        # 'objective': 'binary',
        'metric': ['multi_logloss'],
        # 'metric': "None",
        'learning_rate': 0.002,
        # 'num_leaves': 15,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        # 'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 10,  # Number of bucketed bin for feature values
        # "scale_pos_weight": 8000,
        'nthread': 4,
    }
    stackGBM = singleTraining([nx, ny], [nvx, nvy], cusParams)
    modelName = "_".join(list(skuPair))+"_"+"upper.txt"
    serializer.saveBooster(stackGBM,modelName,stackGBM.best_iteration)

def stackingLogRegression(skuPair,serializer,stackSize,upperTrain,upperTest):
    stackTrain = list()
    stackTest = list()
    for i in range(stackSize):
        modelName = "_".join(list(skuPair)) + "_" + str(i) + ".txt"
        bst, _ = serializer.loadBooster(modelName)
        x, _ = upperTrain
        vx, _ = upperTest
        tmpX = bst.predict(x)
        stackTrain.append(tmpX)
        tmpVX = bst.predict(vx)
        stackTest.append(tmpVX)

    # stackTrainFeature = _transStack(stackTrain)
    x_train = np.concatenate(tuple(stackTrain), axis=1)

    # stackTestFeature = _transStack(stackTest)
    x_test = np.concatenate(tuple(stackTest), axis=1)

    _, y_train = upperTrain
    _, y_test = upperTest

    logistic_regression = LogisticRegression()
    logistic_regression.fit(x_train, y_train)
    ypred_valid = logistic_regression.predict_proba(x_test)
    y_predict_valid = _h_trans_predict(ypred_valid)
    report_valid = classification_report(y_test, y_predict_valid)
    conMatrix_valid = confusion_matrix(y_test, y_predict_valid)
    print (report_valid)
    print (conMatrix_valid)


def trainModel(stageThree,stackNum,serializer,reTrain=False):
    def _de(ds):
        print (len(ds))
        print (len(ds[0]))
        print (len(ds[0][0]))
        print (len(ds[1]))
        print (len(ds[1][0]))

    for skuPair,dataset in stageThree.items():
        print ("")
        print ("<--_+_+_+_+_+_+_+__+_+_+_+_+_+_-->")
        print (skuPair)
        _1_trainDataset,_1_testDataset,_11_trainDataset,_11_testDataset = dataset

        trainDatasets = _dGenerator(_1_trainDataset,stackNum)
        testDatasets  = _dGenerator(_1_testDataset,stackNum)

        stackTrain, upperTrain = trainDatasets[:-1], trainDatasets[-1]
        stackTest, upperTest = testDatasets[:-1], testDatasets[-1]

        for i, stackDs in enumerate(stackTrain):
            modelName = "_".join(list(skuPair))+"_"+str(i)+".txt"
            bst,r = serializer.loadBooster(modelName)
            if r is not None and not reTrain:
                print ("load model %s from disk " % modelName)
                continue

            x,y  = stackDs
            vx,vy = stackTest[i]
            lgbm = singleTraining([x,y],[vx,vy])
            rpath =serializer.saveBooster(lgbm,modelName,lgbm.best_iteration)
            print (rpath)
        stackingLGBM(skuPair,serializer,len(stackTrain),upperTrain,upperTest)
        break

def simpleStacking(weakLevelData, essembleLevelData, serializer,retrainLevelOne=False):
    print ("")
    print ("Start stacking !!")
    basicModels = list()
    essemble_x ,essemble_y = essembleLevelData
    stackingInternal = list()

    for i,wl in enumerate( weakLevelData):
        elementName, train_x,train_y,test_x,test_y = wl
        modelName = "_".join(["levelOne",str(i),elementName])
        print ("")
        tmpBest, t = serializer.loadBooster(modelName)
        if retrainLevelOne or t is None:
            print ("Train level ONE model of %s" % elementName)
            tmpBest = lgbBoosterEearlyStopping(train_x,train_y,test_x,test_y,serializer,modelName)
        else:
            basicModels.append(tmpBest)
        print ("Predict on intermedia level  of size %d " % len(essemble_x))
        internal_x = tmpBest.predict(essemble_x)
        stackingInternal.append(internal_x)
    x_internal = np.concatenate(tuple(stackingInternal), axis=1)
    return [x_internal, essemble_y,basicModels]

def lgbmCV(x_internal,essemble_y):
    # print ("debug")
    params = {
        "objective": "multiclass",
        "boosting": "gbdt",
        "max_depth":-1,
        "num_class": 18,
        "learning_rate": 0.05,
        "min_sum_hessian_in_leaf": 0.002,
        "num_iterations": 5000,
    }
    param_grid = {
        # "num_leaves": [31,127],
        "num_leaves": [31],
        # "bagging_fraction": [0.3,0.9],  # subsample
        "bagging_fraction": [0.3],  # subsample
        # "max_bin":[10,255],
        "max_bin":[10],
        "bagging_seed": [2019],
        # "lambda_l2": [1,2],
        # "lambda_l1": [1,5],
        # "num_iterations": [3000,10000],
    }
    from sklearn.model_selection import GridSearchCV
    estimator = lgb.LGBMClassifier(
        boosting_type=params["boosting"],
        objective=params["objective"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        n_estimators=params["num_iterations"],

    )

    '''
    need dataset for test 
    '''
    gbm = GridSearchCV(estimator, param_grid, cv=3,n_jobs=-1)
    gbm.fit(x_internal,essemble_y)
    print('Best parameters found by grid search are:', gbm.best_params_)
    print (gbm.best_score_)
    print (gbm.cv_results_)

def lgbBoosterEearlyStopping(train_x,train_y,test_x,test_y,serializer,modelName):
    lgbm = singleTraining([train_x,train_y],[test_x,test_y])
    rpath =serializer.saveBooster(lgbm,modelName,lgbm.best_iteration)
    print ("finish training model %s . " % rpath)
    return lgbm

if __name__ == "__main__":
    trainModel()
