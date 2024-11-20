import lightgbm as lgb
from os.path import dirname, realpath
import os
from sklearn.externals import joblib

class Serializer :
    def __init__(self,exePath,folderPath):
        self.__container = folderPath
        self._exepath = exePath

    def _getFolderPath(self,filename):
        rPath = (dirname(self._exepath)) + "/"+self.__container+"/"+filename
        exists = os.path.isfile(rPath)
        return (rPath,exists)

    def loadJoblib(self,fileName):
        rPath,exists = self._getFolderPath(filename=fileName)
        if not exists:
            return ("File %s not exists" % fileName, None)
        ob = joblib.load(rPath)
        return (ob, type(ob))

    def saveJoblib(self,obj,fileName):
        rPath, exists = self._getFolderPath(fileName)
        joblib.dump(obj,rPath)

    def saveBooster(self,clf,fileName,numIter=None):
        rPath , exists = self._getFolderPath(fileName)
        clf.save_model(rPath,num_iteration=numIter)
        return rPath

    def loadBooster(self,fileName):
        rPath , exists = self._getFolderPath(fileName)
        if not exists:
            return ("File %s not exists" % fileName, None)
        bst = lgb.Booster(model_file=rPath)
        return (bst,type(bst))




if __name__ == "__main__":
    testS = Serializer("buffer")
    print (testS._getFolderPath("testFile.txt"))