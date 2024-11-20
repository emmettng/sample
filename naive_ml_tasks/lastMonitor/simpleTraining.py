
from sklearn.externals import joblib
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
        "num_class": 13,
        "bagging_fraction": [0.3],  # subsample
        "feature_fraction": [0.5],  # colsample_bytree
        "bagging_freq": 2,  # subsample_freq
        "bagging_seed": 2019,
        "verbosity": -1,
        "lambda_l2": 2,
        # "lambda_l1": 5,
        # "is_unbalance":True,
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

def _LGB_training(train_data,params,modelName):
    print ("start training process")
    print ("Training name: "+modelName)
    t1 = datetime.datetime.now()
    lgbm=lgb.train(params,train_data)
    t2 = datetime.datetime.now()
    print ("finish training in %s" % (str(t2-t1)))
    return lgbm

def singleTunning(trainData,validData,orgSet):
    EARLY_STOP = True
    # EARLY_STOP = False
    evals_results = {}
    params_generator = _get_params_generator(EARLY_STOP)
    x_train,y_train,x_valid,y_valid = orgSet
    for params in params_generator:
        print(params)
        if not EARLY_STOP:
            lgbm = _LGB_training(trainData, params, "test1")
        else:
            lgbm = lgb.train(params,
                             trainData,
                             # valid_sets=[trainData, validData],
                             # valid_sets=[trainData, testData],
                             # valid_names=['train','valid'],
                             valid_sets=validData,
                             evals_result=evals_results,
                             num_boost_round=200000,
                             early_stopping_rounds=10000,
                             verbose_eval=500,
                             # feval = _wh_eval
                             )

            ypred_valid= lgbm.predict(x_valid,num_iteration=lgbm.best_iteration)
            y_predict_valid = _h_trans_predict(ypred_valid)
            report_valid = classification_report(y_valid,y_predict_valid)
            conMatrix_valid= confusion_matrix(y_valid,y_predict_valid)
            print (report_valid)
            print (conMatrix_valid)

        return lgbm

def _getPhoneSet(stage2set):
    return list(set([r[0] for r in stage2set]))

def _getLabels(stage2set):
    return [r[2] for r in stage2set]

def _getFeatures(stage2set):
    return [r[3:] for r in stage2set]

def trainModel(stageTwo,serializer,retrain=False):
    print ("")
    print ("start the training stage !")
    trainingMemeber = dict()
    for shoe in stageTwo:

        sku,trainSet,validSet = shoe
        print (sku)
        trainPhoneset = _getPhoneSet(trainSet)
        # validPhoneset = _getPhoneSet(validSet)
        X_train = _getFeatures(trainSet)
        Y_train = _getLabels(trainSet)

        X_valid = _getFeatures(validSet)
        Y_valid = _getLabels(validSet)

        nx = np.asarray(X_train)
        ny = np.asarray(Y_train)

        nvx = np.asarray(X_valid)
        nvy = np.asarray(Y_valid)

        trainData = lgb.Dataset(nx,ny)
        validData = lgb.Dataset(nvx,nvy)

        # exists = os.path.isfile(sku+".txt")
        modelName = sku+".txt"
        rpath,exists = serializer._getFolderPath(modelName)
        if exists and not retrain:
            print ("model "+sku+" exist, load from file!")
            trainingMemeber[sku] = trainPhoneset
            continue

        lgbm = singleTunning(trainData,validData,[X_train,Y_train,X_valid,Y_valid])
        modelName = sku+".txt"
        serializer.saveBooster(lgbm,modelName,lgbm.best_iteration)
        # lgbm.save_model(sku +".txt",num_iteration=lgbm.best_iteration)
        trainingMemeber[sku] = trainPhoneset

    return trainingMemeber

def _subSplict(inner_x,inner_y,inner_phone,phoneSet):
    baseLine_X = list()
    baseLine_Y = list()

    target_X = list()
    target_Y = list()
    for i,p in enumerate(inner_phone):
        if p in phoneSet:
            baseLine_X.append(inner_x[i])
            baseLine_Y.append(inner_y[i])
        else:
            target_X.append(inner_x[i])
            target_Y.append(inner_y[i])

    return [baseLine_X,baseLine_Y,target_X,target_Y]

def _predictEva(localModel,x,y_true):
    ypred_valid = localModel.predict(x)
    y_predict_valid = _h_trans_predict(ypred_valid)
    report_valid = classification_report(y_true, y_predict_valid)
    conMatrix_valid = confusion_matrix(y_true,y_predict_valid)
    print(report_valid)
    print(conMatrix_valid)
    print("")

def compare(serializer,stageTwo,trainingMember):
    for sku,phoneSet in trainingMember.items():
        print ("")
        print ("start evaluate stage of "+sku)
        modelName = sku+".txt"
        bst,t = serializer.loadBooster(modelName)
        # bst = lgb.Booster(model_file=sku+".txt")
        for shoe in stageTwo:

            innersku,innertrainSet,innervalidSet = shoe
            if innersku == sku: continue
            print ("   evaluate at "+innersku)

            trainPhoneset = _getPhoneSet(innertrainSet)
            validPhoneset = _getPhoneSet(innervalidSet)

            X_train_tmp = _getFeatures(innertrainSet)
            Y_train_tmp = _getLabels(innertrainSet)

            X_valid_tmp = _getFeatures(innervalidSet)
            Y_valid_tmp = _getLabels(innervalidSet)

            inner_x = X_train_tmp+X_valid_tmp
            inner_y = Y_train_tmp+Y_valid_tmp
            innerPhoneset = trainPhoneset+validPhoneset

            baseLine_X,baseLine_Y,target_X,target_Y = _subSplict(inner_x,inner_y,innerPhoneset,phoneSet)
            _predictEva(bst,baseLine_X,baseLine_Y)
            _predictEva(bst,target_X,target_Y)

def inspectModels(stageTwo,footFeatures,shoeFeatures):
    composeOne = footFeatures+footFeatures+shoeFeatures[1:]
    seg1 = len(footFeatures)
    seg2 = len(shoeFeatures[1:])
    for shoe in stageTwo:
        innersku,innertrainSet,innervalidSet = shoe
        bst = lgb.Booster(model_file=innersku+".txt")
        feature_importance = bst.feature_importance()
        print (innersku)
        print (feature_importance)
        print (len(feature_importance))
        print (len(composeOne))
        print (list(zip(feature_importance,composeOne)))
        print (list(zip(feature_importance,composeOne))[:seg1])
        print (list(zip(feature_importance,composeOne))[seg1:seg1*2])
        print (list(zip(feature_importance,composeOne))[seg1*2:])
        break


if __name__ == "__main__":
    trainModel()
