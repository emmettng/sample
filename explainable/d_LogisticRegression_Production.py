from a_preprocess import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib

GRIDMODEL = "grid_clf.pkl"


'''
type:           String
                -> [String]                                         feature list of critical features.
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
                elif explain['coef'].values[i] > 0:
                    colors.append('g')
                else:
                    colors.append('r')
                tmpExplain['coef'].plot(kind='bar', color=colors)
                plt.savefig(ex_id)
        r = lr.predict_proba(oncoded)
        rd = lr.predict(oncoded)
        return r,rd
    return naiveCurry

if __name__== "__main__":
    model_root = "modelPck_training_test_lr_MinMax_recall_score/"
    positiveFeatures = joblib.load(model_root+"Features_positive.pkl")
    nagativeFeatures = joblib.load(model_root+"Features_nagative.pkl")
    effectiveFeatures = pd.concat([positiveFeatures,nagativeFeatures])

    testDict = joblib.load("./dataSource/lrTestSample.pkl")

    singlePredict = _naiveBatchProduction(model_root,explain=effectiveFeatures)
    for k,vs in testDict.items():
        for iv,v in enumerate(vs):
            r ,rd = singlePredict(v,1,model_root+"_pred_"+(k)+"_"+str(iv)+".png")
            break



