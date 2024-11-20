from a_preprocess import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib

GRIDMODEL = "grid_clf.pkl"
LABELENCODER = "LabelEncoder.pkl"



'''
type:           String
                -> (DataFrame -> (Float,Int)
description:    return the predict function.                        being used for interpretation when changing value of key features.
'''
def _naiveBatchProduction(model_root,explain=None):
    def naiveCurry (features_input,num, ex_id=None):
        lr = joblib.load(model_root+GRIDMODEL)
        lenc = joblib.load(model_root+"LabelEncoder.pkl")
        ohenc = joblib.load(model_root+"OneHotEncoder.pkl")
        scale = joblib.load(model_root+"MinMaxScaler.pkl")

        labelcoded = map(featureCoding(lenc),features_input.values[:num])
        oncoded= map(featureCoding(ohenc),labelcoded)
        oncoded = map(np.concatenate,oncoded)
        oncoded = list(oncoded)
        oncoded = scale.transform(oncoded)

        if (explain is not None) and (num==1) and (ex_id is not None): ## only save explination when processing single input
            tmpExplain = copy.deepcopy(explain)
            obs = oncoded[0]
            colors = list()
            for i, row in enumerate(explain.index.values):
                if obs[row] == 0:
                    colors.append('lightgrey')
                elif explain['importances'].values[i] > 0:
                    colors.append('g')
                else:
                    colors.append('r')
                tmpExplain['importances'].plot(kind='bar', color=colors)
                plt.savefig(ex_id+ "_pred.png")

        r = lr.predict_proba(oncoded)
        rd = lr.predict(oncoded)

        pathList = list()
        for i,e in enumerate(lr.best_estimator_.estimators_):
            path = e.decision_path(oncoded)
            path = path.toarray()[0]
            nodes = [i for i,v in enumerate(path) if v==1]
            treeArray = (e.tree_.__getstate__()['nodes'])
            pathFeatures = list()
            for step in nodes[:-1]:
                for n in treeArray:
                    ln = n[0]
                    if step < ln:
                        state = n[2]
                        observation = oncoded[0][state]
                        pathFeatures.append((state,observation))
                        break
                    else: continue
            pathFeatures.append(nodes[-1])
            pathList.append(pathFeatures)
        return r,rd, pathList
    return naiveCurry

if __name__== "__main__":
    model_root = "modelPck_training_test_rf_MinMax_recall_score/"
    effectiveFeatures= joblib.load(model_root+"Features_effective.pkl")
    otherFeatures= joblib.load(model_root+"Features_other.pkl")

    testDict = joblib.load("./dataSource/lrTestSample.pkl")

    singlePredict = _naiveBatchProduction(model_root)

    for k,vs in testDict.items():
        for iv,v in enumerate(vs):
            r,d, pathList =  singlePredict(v,1,model_root+(k)+"_"+str(iv))
            print ("The prediction of record %s is %d" % (iv,d,))
            print ("Path List below:")
            for p in pathList:
                print (p)
            print ("")
        break
