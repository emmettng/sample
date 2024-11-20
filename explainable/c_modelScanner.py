from b_training import *

import numpy as np
import umap
from sklearn.externals import joblib

def features4LogisticRegression(folderPath):
    modelPath = folderPath+GRIDMODEL
    featuresPath = folderPath+"features.pkl"

    lr = joblib.load(modelPath)
    features = joblib.load(featuresPath)
    coefs = lr.best_estimator_.coef_[0]
    tmp = list(zip(features,coefs))
    coefDF = pd.DataFrame(tmp,columns=['features','coef'])
    sortedDF = coefDF.sort_values(by=['coef'])
    positiveFeatures= sortedDF[sortedDF['coef']>0].sort_values(by=['coef'],ascending=[False])
    nagativeFeatures= sortedDF[sortedDF['coef']<0].sort_values(by=['coef'],ascending=[False])
    otherFeatures = sortedDF[sortedDF['coef']==0]

    frames = [positiveFeatures,nagativeFeatures]
    effectFeatures = pd.concat(frames)
    effectFeatures['coef'].plot(kind='bar',color=np.where(effectFeatures['coef']>0,'g','r'))
    plt.savefig(folderPath+"effetiveFeatures.png")
    Save2pkl(positiveFeatures,folderPath+"Features_positive.pkl")
    Save2pkl(nagativeFeatures,folderPath+"Features_nagative.pkl")
    Save2pkl(otherFeatures, folderPath+ "Features_other.pkl")
    print(effectFeatures)
    return positiveFeatures,nagativeFeatures,otherFeatures

def features4RandomForest(folderPath):
    modelPath = folderPath+GRIDMODEL
    featuresPath = folderPath+"features.pkl"

    rf = joblib.load(modelPath)
    features = joblib.load(featuresPath)
    importances = rf.best_estimator_.feature_importances_
    print (importances)
    print (len(importances))
    print (features)
    tmp = list(zip(features,importances))
    featureW= pd.DataFrame(tmp,columns=['features','importances'])
    print (featureW)
    sortedDF = featureW.sort_values(by=['importances'],ascending=[False])
    print (sortedDF)

    importantFeatures= sortedDF[sortedDF['importances']!=0]
    otherFeatures = sortedDF[sortedDF['importances']==0]
    print (importantFeatures)

    Save2pkl(otherFeatures, folderPath+ "Features_other.pkl")
    Save2pkl(importantFeatures, folderPath+ "Features_effective.pkl")

    importantFeatures['importances'].plot(kind='bar',color=['g'])
    plt.savefig(folderPath+"importantFeatures.png")

    return importantFeatures, otherFeatures

def umapTransform(folderPath):
    Xpath = folderPath+"X.pkl"
    ypath = folderPath+"y.pkl"
    X = joblib.load(Xpath)
    y = joblib.load(ypath)

    nX = X.values[:5000]
    ny = y.values[:5000]

    metricList = ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule', 'wminkowski']
    n_neighbor_list = [15,20,40,60,80,100,200]
    ry = ny.reshape([-1,])
    min_dist=0.3

    for metric in metricList:
        for n_neighbors in n_neighbor_list:
            trans = umap.UMAP(metric=metric,min_dist=min_dist, n_neighbors = n_neighbors,n_epochs=200).fit_transform(nX,ry)
            plt.scatter(trans[:,0],trans[:,1],s=5,c=np.where(ry==1,'g','r'))
            t = ("metric: %s, n_neighbors: %d, min_dist: %f" % (metric,n_neighbors,min_dist))
            plt.title(t)
            plt.savefig("./umap/"+t+".png")
            plt.close()


if __name__ == "__main__":

    '''
    1. Save effective features of logistic regression
    '''
    lr_model_root= "modelPck_training_test_lr_MinMax_recall_score/"
    features4LogisticRegression(lr_model_root)

    '''
    2. Select important features of random forest.
    '''
    # rf_model_root = "modelPck_training_test_rf_MinMax_recall_score/"
    # features4RandomForest(rf_model_root)

    '''
    3. Supervised dimension reduction for identifying decision boundary.
    This well take a looooooog time.
    '''
    # umapTransform(rf_model_root)


