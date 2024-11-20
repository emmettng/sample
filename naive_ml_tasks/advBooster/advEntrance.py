import itertools
from numpy.linalg import norm
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import xlsxwriter


from wh.basicProcessor.v1process import *
from wh.basicProcessor.v8process import *
from wh.basicProcessor.featureInfor import *
from wh.basicProcessor.FootProcess import *
from wh.basicProcessor.ShoeProcess import *
from wh.basicProcessor.serializer import *

from wh.advBooster.boostWorkers import *

from wh._0_Meta.dataSource import *


def _computeStep(vlist):
    """
    :param vlist: [(size,[features])...]
    :return: [feature steps]
    """
    tlist = zip(vlist[:-1],vlist[1:])
    diffList = list()
    for vs in tlist:
        lower,upper = vs[0][1], vs[1][1]
        diff = np.asarray(upper) - np.asarray(lower)
        diffList.append(diff)
    diffMean = np.mean(diffList,axis=0)
    return diffMean

def _newFeatures(standar,steps,segLength):
    beg = 0
    end = segLength
    features = list()
    while(beg<len(standar)):
        tmpStandard = standar[beg:end]
        tmpSteps = steps[beg:end]
        features.append(norm(tmpStandard))
        features.append(norm(tmpSteps))
        beg+=segLength
        end+=segLength
        if end >len(standar):
           end = len(standar)
    return features



def _shoeProcessor(shoeVer,shoeFeatures,segLength=1):
    shoeV = SHOE_DICT.get(shoeVer,None)
    s4 = getShoeFeatures(shoeV,shoeFeatures,True)

    stepDict = dict()
    standarDict = dict()
    newFeatureDict = dict()

    comDict = dict()
    for s in s4:
        sku,size,gender = s[0],s[1],s[2]
        features = list(map(lambda x:float(x),s[3:]))
        tmpList = comDict.get(sku,list())
        tmpList.append((int(size),features))
        comDict[sku] = tmpList
        if gender == "2" and size == "230":
            standarDict[sku]=features
        if gender == "1" and size == "250":
            standarDict[sku]=features
    for sku,vlist in comDict.items():
        vlist.sort(key=lambda tup: tup[0])
        steps = _computeStep(vlist)
        stepDict[sku] = steps

    for sku,standar in standarDict.items():
        steps = stepDict.get(sku)
        newFeatures = _newFeatures(standar,steps,segLength)
        newFeatureDict[sku]=newFeatures
    return newFeatureDict

def _retrieveDatasets(resVer,footVer,footFeatures,shoeVer,shoeFeatures,gender):
    '''
    resDict = {"v1":Response_v1,"v81":Response_v8_1,"v82":Response_v8_2}
    shoeDict = {"shoeClassical":ShoesClassical,"shoeCalibrate":ShoesCalibrate,"shoeAll04":ShoesClassicalAll_04,"shoeAll05":ShoesClassicalAll_05}
    footDict = {"foot01":FootClassical_01,"foot04":FootClassical_04,"foot05":FootClassical_05}

    :param resVer:
    :param footVer:
    :param shoeVer:
    :param gender:
    :return:
    '''
    ## --- get file path of all these data ---------
    resV = RES_DICT.get(resVer,None)
    footV = FOOT_DICT.get(footVer,None)
    shoeV = SHOE_DICT.get(shoeVer,None)
    ###--- get file path of all these data ---------###

    if resV is None or footV is None or shoeV is None:
        return ("wtf")

    if resVer == "v1":
        res = getSizeV1(resV)
    else:
        res = getSizeV8(resV)

    # footFeatures = classicalFoot_v1["contFeatures"]+classicalFoot_v1["orderedDisc"]
    footFeaturesLeft = list(map(lambda x:x+"_left",footFeatures))
    footFeaturesRight= list(map(lambda x:x+"_right",footFeatures))
    lff = getFooFeatures(footV,footFeaturesLeft,True)
    rff = getFooFeatures(footV,footFeaturesRight,True)

    lfDict = {r[0]: r[1:] for r in lff}
    rfDict = {r[0]: r[1:] for r in rff}

    # shoeFeatures0 = classicalShoe["contFeatures"]+ classicalShoe["orderedDisc"] + classicalShoe["unorderedDisc"]
    # shoeFeatures4 = ["gender"]+classicalShoe["contFeatures"]
    s4 = getShoeFeatures(shoeV,shoeFeatures,True)

    # sDict = { (s[0],s[1]): s[3:] for s in s4 if s[2] == gender}

    if gender == "1":
        sDict = {s[0]:s[3:] for s in s4 if s[2] == gender and s[1]=="250"}
    else:
        sDict = {s[0]:s[3:] for s in s4 if s[2] == gender and s[1]=="230"}


    stageOne = list()
    for r in res:
        phone,sku,size = r

        size_code = SIZE_RANGE.index(int(size))
        r[2] = size_code

        lfv = lfDict.get(phone,None)
        rfv = rfDict.get(phone,None)
        # sfv = sDict.get((sku,size),None)
        sfv = sDict.get(sku,None)
        # snfv = engShoeFeatures.get(sku,None)

        if lfv is None or rfv is None or sfv is None :
            continue
        lfvNum = list(map(lambda x:float(x),lfv))
        rfvNum = list(map(lambda x:float(x),rfv))
        sfvNum = list(map(lambda x:float(x),sfv))

        stageOne.append(r+lfvNum+rfvNum+sfvNum)
        # stageOne.append(r+rfvNum+sfvNum)

    return stageOne

def _get_meta_data(filename):
    filepath = realpath(__file__)
    rPath = (dirname(filepath)) + "/meta/" + filename
    return rPath

def split4Monitor(stage1,proportion):
    skuSet = list(set([r[1] for r in stage1]))
    phoneSet = list(set([r[0] for r in stage1]))

    # if gender == "1":
    #     size_range = joblib.load(_get_meta_data(MALE_SIZE_RANGE_FILE_NAME))
    # else:
    #     size_range = joblib.load(_get_meta_data(FEMALE_SIZE_RANGE_FILE_NAME))
    size_range = SIZE_RANGE

    stageTwo = list()
    for sku in skuSet:
        trainingSet = list()
        validSet = list()

        phone_sample_num = round(proportion * len(phoneSet))
        phone_training_set= random.sample(phoneSet, phone_sample_num)

        for record in stage1:
            inner_phone,inner_sku,inner_size = record[0],record[1],record[2]
            if inner_sku != sku : continue
            size_code = size_range.index(int(inner_size))
            record[2] = size_code
            if inner_phone in phone_training_set:
                trainingSet.append(record)
            else:
                validSet.append(record)
        stageTwo.append((sku,trainingSet,validSet))

    return stageTwo

# def stageFour(stage2,stage3):
    #---------------------- Stage 4 -------------------------------------------------###
    # exePath = realpath(__file__)
    # serilizer = Serializer(exePath,"buffer")
    # compare(serilizer,stage2,stage3)

def stageTraining(stage3,stackNum,retain=False):
    ##---------------------- Stage 3 -------------------------------------------------###
    print ("")
    exePath = realpath(__file__)
    serilizer = Serializer(exePath,"models")
    stage4 = trainModel(stage3,stackNum,serilizer,retain)
    return stage4

def stackSKU(stageThree):
    print ("")
    basicLevel= list()
    essembX = list()
    essembY = list()
    for skuPair,dataset in stageThree.items():
        print ("")
        print ("<--_+_+_+_+_+_stack sku data preparetion_+_+_+_+_-->")
        print (skuPair)
        _1_trainDataset,_1_testDataset,_11_trainDataset,_11_testDataset = dataset
        print ("Training set: %d " % len(_1_trainDataset[0]))
        print ("Testing set: %d " % len(_1_testDataset[0]))
        print ("valid leak to test set: %d " % len(_11_trainDataset[0]))
        print ("valid leak to train set: %d " % len(_11_testDataset[0]))
        basicLevel.append([str(skuPair)] + _1_trainDataset + _1_testDataset)
        essembX += _11_trainDataset[0]
        essembX += _11_testDataset[0]
        essembY += _11_trainDataset[1]
        essembY += _11_testDataset[1]

    return basicLevel,[essembX,essembY]

def stackingEntrance(firstLeve,sndLevel):
    print ("")
    exePath = realpath(__file__)
    serilizer = Serializer(exePath,"models")
    x_internal, essemble_y, basicModels= simpleStacking(firstLeve,sndLevel,serilizer)
    return [x_internal,essemble_y,basicModels]

def _getLabels(stage2set):
    return [r[2] for r in stage2set]

def _getFeatures(stage2set):
    return [r[3:] for r in stage2set]

def _getDataset(stage2set):
    x = _getFeatures(stage2set)
    y = _getLabels(stage2set)
    return [x,y]

def stageThree (stage2,phoneProportion,skuProportion,stageMarker,recalculate=True):

    def _concate(skulist,dDict):
        rlist = list()
        for s in skulist:
            d = dDict.get(s)
            rlist+=d
        return rlist

    print ("")
    print ("Start Stage Three at %s !!!" % stageMarker.get("resVersion"))
    stageName = list(map(lambda x: str(x), stageMarker.values()))
    tmpFileName = "_".join(["stageThree"] + stageName)

    print ("")
    exePath = realpath(__file__)
    serilizerBuffer = Serializer(exePath,"buffer")
    stageThree, t = serilizerBuffer.loadJoblib(tmpFileName)
    if t is not None and (not recalculate):
        print ("Load stageThree %s from disk" % tmpFileName)
        [print (k) for k in stageThree.keys()]
        print ("")
        return stageThree

    print ("")
    print ("Process stage 3 from csv!")
    print ("Stage three involves sku pair:")
    skuSet = [r[0] for r in stage2]
    dataDict = {r[0]:r[1]+r[2] for r in stage2}

    r = int((1-skuProportion)*len(skuSet))
    combs= list(itertools.combinations(skuSet,r))
    datasetDict = dict()
    cnt = 0
    for cb in combs:
        print (cb)
        cnt +=1
        if cnt >40:
            break
        testSKU = list(cb)
        trainSKU = [s for s in skuSet if s not in testSKU]
        _0_testDataset = _concate(testSKU,dataDict)
        _0_trainDataset = _concate(trainSKU,dataDict)

        trainPhoneset = list(set([r[0] for r in _0_trainDataset]))
        phone_sample_num = round(phoneProportion* len(trainPhoneset))
        phone_training_set= random.sample(trainPhoneset, phone_sample_num)

        _1_trainDataset = _getDataset([t for t in _0_trainDataset if t[0] in phone_training_set])
        _1_testDataset =  _getDataset([t for t in _0_testDataset if t[0] not in phone_training_set])

        _11_trainDataset = _getDataset([t for t in _0_trainDataset if t[0] not in phone_training_set])
        _11_testDataset =  _getDataset([t for t in _0_testDataset if t[0] in phone_training_set])

        datasetDict[cb] = [_1_trainDataset,_1_testDataset,_11_trainDataset,_11_testDataset]


    print ("%d sku pairs" % len(datasetDict.keys()))
    serilizerBuffer.saveJoblib(datasetDict,tmpFileName)
    return datasetDict


def stageTwoPhoneSKU(stage1,stageMarker):
    ##------------------------- Stage 2 ----------------------------------------------###
    '''
    :param stage1:
    :param stageMarker:
    :return:

    '''
    phoneProportion = stageMarker.get("phoneProportion")
    stage2 = split4Monitor(stage1,phoneProportion)
    '''
    (sku,traingSet,validSet)
    trainSet , validSet ::[[phone,sku,size,....]]
    '''
    print ("")
    print ("Stage Two")
    print ("relate num of sku: %d " % len(stage2))
    print ("Record dimension of each sku: %d " % len(stage2[0]))
    print ("number of training reocrds: %d "% len(stage2[0][1]))
    print ("number of validation records: %d " % len(stage2[0][2]))
    print (stage2[0][1][0])
    return stage2

def stageShoeFeatures(shoeFeatures,stageMarker):
    def _decodeShoeFeatures(fDict):
        for k,v in fDict.items():
            print (k)
            print (v)
            break

    shoeVer = stageMarker.get("shoeVersion")
    newShoeFeatureSeg = stageMarker.get("shoeFeatureSeg")
    stageName = list(map(lambda x:str(x),stageMarker.values()))
    tmpFileName = "_".join(["stageShoeFeatures"]+stageName)

    print ("")
    exePath = realpath(__file__)
    serilizerBuffer = Serializer(exePath,"buffer")
    stageShoeFeature, t = serilizerBuffer.loadJoblib(tmpFileName)
    if t is not None:
        print ("load shoeFeatures %s from disk" % tmpFileName)
        print ("")
        _decodeShoeFeatures(stageShoeFeature)
        return stageShoeFeature
    stageShoeFeature= _shoeProcessor(shoeVer,shoeFeatures,newShoeFeatureSeg)
    serilizerBuffer.saveJoblib(stageShoeFeature,tmpFileName)
    print ("")
    print ("Process stage shoe Features from csv")
    _decodeShoeFeatures(stageShoeFeature)
    return stageShoeFeature

def stackingLayerTwo(internal_x,essemble_y):
    lgbmCV(internal_x,essemble_y)

def _stageProcessor(stageMarker,stageList,stageKeys):
    markerStr = list(map(lambda x:str(x),stageMarker.values()))
    listLenStr = str(len(stageList))
    stageName = "_".join(stageKeys+[listLenStr]+markerStr)
    exePath = realpath(__file__)
    serializer = Serializer(exePath,"buffer")
    stage, t = serializer.loadJoblib(stageName)
    return serializer, stageName, stage, t


def stageOne(gender,footFeatures,shoeFeatures,stageMarker):
    ##------------------Stage 1 ---------------------------------------------------###
    '''
    :param gender:
    :param footFeatures:
    :param shoeFeatures:
    :param stageMarker:
    :return:
        [[phone,sku,size,....]...]
    '''
    def _decodeStage1(stage1):
        print ("Stage One length: %d" % len(stage1))
        print (stage1[0])
        print (len(stage1[0]))
        print (stage1[1])

    resVer = stageMarker.get("resVersion")
    footVer = stageMarker.get("footVersion")
    shoeVer = stageMarker.get("shoeVersion")

    serilizerBuffer, stageName,stage1, t = _stageProcessor(stageMarker,footFeatures+shoeFeatures,["stageOne"])
    if t is not None:
        print ("load stage1 %s from disk" % stageName)
        _decodeStage1(stage1)
        print ("")
        return stage1

    print ("")
    print ("Process stage one from csv")
    stage1 = _retrieveDatasets(resVer,footVer,footFeatures,shoeVer,shoeFeatures,gender)
    serilizerBuffer.saveJoblib(stage1,stageName)
    _decodeStage1(stage1)
    return stage1

def _loadModels(level):
    exePath = realpath(__file__)
    modelFolder= (dirname(exePath)) + "/models"
    serilizerModels = Serializer(exePath, "models")
    print (modelFolder)
    modelList = list()
    for root, dirs, mfiles in os.walk(modelFolder,topdown=False):
        print (mfiles)

    for m in range(len(mfiles)):
        for f in mfiles:
            l = f.split("_")[0]
            num = int(f.split("_")[1])
            if m == num and l == level:
                modelList.append(serilizerModels.loadBooster(f)[0])

    return modelList

def _anaModelName(level):
    exePath = realpath(__file__)
    modelFolder= (dirname(exePath)) + "/models"
    serilizerModels = Serializer(exePath, "models")
    modelList = list()
    tfiles = None
    for root, dirs, mfiles in os.walk(modelFolder,topdown=False):
        tfiles=mfiles

    skuList = list()
    skuDict = dict()
    for m in range(len(tfiles)):
        for f in tfiles:
            l = f.split("_")[0]
            num = int(f.split("_")[1])
            if m == num and l == level:
                skuTMPs1 = f.split("_")[2]
                skuTMPs2= skuTMPs1.split("'")
                skuTMPs3 = [s for s in skuTMPs2 if len(s) == 14]
                skuList+= skuTMPs3
    for sku in skuList:
        cnt = skuDict.get(sku,0)
        cnt+=1
        skuDict[sku] = cnt
    return skuDict

def intermediaPred(basicModels,essemble_x):

    stackingInternal = list()
    for i,tmpBest in enumerate(basicModels):
        print ("Start basic model %d " % i)
        internal_x = tmpBest.predict(essemble_x)
        stackingInternal.append(internal_x)
    x_internal = np.concatenate(tuple(stackingInternal), axis=1)
    return x_internal

def _inspectLevelTwo(level2model,testSet):
    def _ttF(t):
        if t[0] == t[1]:
            return 1
        else:
            return 0

    test_x,test_y = testSet
    p_y= level2model.predict(test_x)
    pred_y = list(map(lambda x:list(x).index(np.max(x)),p_y))
    accu = accuracy_score(test_y,pred_y)
    ttlist = list(map(_ttF,list(zip(test_y,pred_y))))
    return [pred_y,accu,ttlist]


def js(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
   # normalize
    p /= p.sum()
    q /= q.sum()
    m = (p + q) / 2
    return (entropy(p, m) + entropy(q, m)) / 2

def _getSKUcombs(stage,skuProportion):
    skuSet = list(set([r[1] for r in stage]))
    r = int((1 - skuProportion) * len(skuSet))
    combs = list(itertools.combinations(skuSet, r))
    return combs

def _normalTrans(stage1,stageMarker):
    serializer, stageName, stage, t  = _stageProcessor(stageMarker,[],["normalTransform"])
    if t is not None:
        idList,dataSet = stage
        return [idList,dataSet]

    idList = list()
    for record in stage1:
        inner_phone,inner_sku,inner_size = record[0],record[1],record[2]
        idList.append([inner_phone,inner_sku,inner_size])
    dataSet = _getDataset(stage1)
    serializer.saveJoblib([idList,dataSet],stageName)
    return [idList,dataSet]



def _normalSplitPhoneSKU(stage1,sku_test_set,stageMarker):
    phoneProportion = stageMarker["phoneProportion"]

    skuSet = list(set([r[1] for r in stage1]))
    phoneSet = list(set([r[0] for r in stage1]))

    train= list()
    test= list()
    trainIDs = list()
    testIDs = list()

    phone_sample_num = round(phoneProportion * len(phoneSet))
    phone_training_set = random.sample(phoneSet, phone_sample_num)

    sku_training_set = [s for s in skuSet if s not in sku_test_set]

    serializer, stageName, stage, t  = _stageProcessor(stageMarker,sku_training_set,["normalSplit"]+[str(sku_test_set)])
    if t is not None:
        [trainIDs,testIDs, trainSets, testSets] = stage
        return [trainIDs,testIDs, trainSets, testSets,sku_test_set]
    for record in stage1:
        inner_phone,inner_sku,inner_size = record[0],record[1],record[2]
        if inner_phone in phone_training_set and inner_sku in sku_training_set:
            trainIDs.append([inner_phone,inner_sku,inner_size])
            train.append(record)
        elif inner_phone not in phone_training_set and inner_sku not in sku_training_set:
            testIDs.append([inner_phone,inner_sku,inner_size])
            test.append(record)
        else:
            continue
    print ("")
    print ("<---------------->")
    print ("Normal split stage: ")
    print ("%d number of record from stage1 need to be split." % len(stage1))
    print (" %d number of sku in reocrd " % len(skuSet))
    print (" %d number of phone in record " % len(phoneSet))
    print (" %d number of phone will be in training set." % len(phone_training_set))
    print (" %d number of sku will be in training set. " % len(sku_training_set))
    print (" training size: %d " % len(train))
    print (" testing  size: %d " % len(test))
    print ("Test sku includes %s " % str(sku_test_set))
    trainSets = _getDataset(train)
    testSets = _getDataset(test)
    serializer.saveJoblib([trainIDs,testIDs, trainSets, testSets],stageName)
    return [trainIDs,testIDs, trainSets, testSets,sku_test_set]

def _normalSplitPhone(stage1,stageMarker):
    phoneProportion = stageMarker["phoneProportion"]
    phoneSet = list(set([r[0] for r in stage1]))
    skuSet = list(set([r[1] for r in stage1]))

    train= list()
    test= list()
    trainIDs = list()
    testIDs = list()

    phone_sample_num = round(phoneProportion * len(phoneSet))
    phone_training_set = random.sample(phoneSet, phone_sample_num)

    serializer, stageName, stage, t  = _stageProcessor(stageMarker,[],["normalSplitPhone"])
    if t is not None:
        [trainIDs,testIDs, trainSets, testSets] = stage
        return [trainIDs,testIDs, trainSets, testSets]
    for record in stage1:
        inner_phone,inner_sku,inner_size = record[0],record[1],record[2]
        if inner_phone in phone_training_set :
            trainIDs.append([inner_phone,inner_sku,inner_size])
            train.append(record)
        elif inner_phone not in phone_training_set :
            testIDs.append([inner_phone,inner_sku,inner_size])
            test.append(record)
        else:
            continue
    print ("")
    print ("<------Phone SPLIT ONLY-------->")
    print ("Normal split \"Phone\" : ")
    print ("%d number of record from stage1 need to be split." % len(stage1))
    print (" %d number of sku in reocrd " % len(skuSet))
    print (" %d number of phone in record " % len(phoneSet))
    print (" %d number of phone in training set. " % len(phone_training_set))
    print (" training size: %d " % len(train))
    print (" testing  size: %d " % len(test))
    trainSets = _getDataset(train)
    testSets = _getDataset(test)
    serializer.saveJoblib([trainIDs,testIDs, trainSets, testSets],stageName)
    return [trainIDs,testIDs, trainSets, testSets]

def _normalSplitSKU(stage1,sku_test_set,stageMarker):
    '''stage1 ::  [[phone,sku,size,....]...] '''
    skuSet = list(set([r[1] for r in stage1]))
    phoneSet = list(set([r[0] for r in stage1]))
    # size_range = SIZE_RANGE

    train= list()
    test= list()
    trainIDs = list()
    testIDs = list()


    sku_training_set = [s for s in skuSet if s not in sku_test_set]

    serializer, stageName, stage, t  = _stageProcessor(stageMarker,sku_training_set,["normalSplitSKU"]+[str(sku_test_set)])
    if t is not None:
        [trainIDs,testIDs, trainSets, testSets] = stage
        return [trainIDs,testIDs, trainSets, testSets,sku_test_set]
    for record in stage1:
        inner_phone,inner_sku,inner_size = record[0],record[1],record[2]
        if inner_sku in sku_training_set:
            trainIDs.append([inner_phone,inner_sku,inner_size])
            train.append(record)
        elif inner_sku not in sku_training_set:
            testIDs.append([inner_phone,inner_sku,inner_size])
            test.append(record)
        else:
            continue
    print ("")
    print ("<------SKU SPLIT ONLY-------->")
    print ("Normal split \"SKU\" : ")
    print ("%d number of record from stage1 need to be split." % len(stage1))
    print (" %d number of sku in reocrd " % len(skuSet))
    print (" %d number of phone in record " % len(phoneSet))
    print (" %d number of sku in training set. " % len(sku_training_set))
    print (" training size: %d " % len(train))
    print (" testing  size: %d " % len(test))
    print ("Test sku includes %s " % str(sku_test_set))
    trainSets = _getDataset(train)
    testSets = _getDataset(test)
    serializer.saveJoblib([trainIDs,testIDs, trainSets, testSets],stageName)
    return [trainIDs,testIDs, trainSets, testSets,sku_test_set]

def _inspectModels(levelNmodels,x,y_true):
    reportDict = dict()
    for mN, l2m in enumerate(levelNmodels):
        print("Test on level 2 of model: %s " % mN)
        predList, accu, ttlist = _inspectLevelTwo(l2m, [x, y_true])
        reportDict[mN] = [predList, accu, ttlist]
    return reportDict


def _loadBoosters(modelname=None):
    exePath = realpath(__file__)
    serilierModel = Serializer(exePath,"models")
    modelFolder= (dirname(exePath)) + "/models"

    boosters = dict()
    for root, dirs, mfiles in os.walk(modelFolder,topdown=False):
        print (mfiles)

    for f in mfiles:
        ind = f.split("_")[-1]
        if modelname is not None:
            n = f.split("_")[0]
            if n != modelname:
                continue
        bst = serilierModel.loadBooster(f)[0]
        yield (ind,bst)

def _singleModelCheck(bst, dataset,checkID):
    x,y = dataset
    p_y= bst.predict(x)
    pred_y = list(map(lambda x:list(x).index(np.max(x)),p_y))
    report_valid = classification_report(y,pred_y)
    conMatrix_valid= confusion_matrix(y,pred_y)
    print ("")
    print ("---single Check---")
    print (checkID)
    print (report_valid)
    print (conMatrix_valid)
    return pred_y

def _modelscheckDict(dataset,stageMarker,modelname = None):
    serializer, stageName,check_dict,t =  _stageProcessor(stageMarker,dataset[0],["normalTransform_v81"])
    if t is not None:
        return check_dict
    boostGenerator= _loadBoosters(modelname)
    v81_check_dict =dict()
    for bg in boostGenerator:
        ind,bst = bg
        x,y = dataset
        p_y= bst.predict(x)
        pred_y = list(map(lambda x:list(x).index(np.max(x)),p_y))
        report_valid = classification_report(y,pred_y)
        conMatrix_valid= confusion_matrix(y,pred_y)
        print ("")
        print (ind)
        print (report_valid)
        print (conMatrix_valid)
        v81_check_dict[ind] = pred_y
    # serializer.saveJoblib(v81_check_dict,stageName)
    return v81_check_dict

def _ttF(t):
    if t[0] == t[1]:
        return 1
    else:
        return 0

def _ttFSmaller(t):
    if t[0] < t[1]:
        return 1
    else:
        return 0

def __computeCorrelation(reportArray,fileName):
    correArray = np.zeros([len(reportArray),len(reportArray)])
    for i , r in enumerate(reportArray):
        for innerI, innerR in enumerate(reportArray):
            correArray[i][innerI] = pearsonr(r,innerR)[0]
    workbook = xlsxwriter.Workbook(fileName+'.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    for col, data in enumerate(correArray):
       worksheet.write_column(row,col,data)
    workbook.close()
    corMean = np.mean(correArray)
    corStd = np.std(correArray)
    return corMean,corStd


def _iterAna(yList,pDict,name):
    reportArray = list()
    verifyArray = list()
    for k, p in pDict.items():
        ttlist = list(map(_ttF,list(zip(yList,p))))
        reportArray.append(p)
        verifyArray.append(ttlist)

    corMean,corStd = __computeCorrelation(reportArray,name)
    return corMean,corStd



def iterOverSKUcomb(trainingStage,testingDS,trainingStageMarker,splitRange):

    combLengthCum = list()
    exePath = realpath(__file__)
    serilierModel = Serializer(exePath,"models")
    serilizerBuffer = Serializer(exePath,"buffer")

    corAnaDict = dict()

    for splitPor in splitRange:
        trainingStageMarker["skuProportion"] = splitPor
        skuCombsTesting= _getSKUcombs(trainingStage, trainingStageMarker["skuProportion"])
        if len(skuCombsTesting[0]) in combLengthCum: continue
        combLengthCum.append(len(skuCombsTesting[0]))
        print ("")
        print("====<<=====<<==== split over sku and phone ====>>=====>>>=====")
        print ("+++ Test sku comb numbers : %d" % len(skuCombsTesting))
        print ("+++ Each TEST sku length: %d" % len(skuCombsTesting[0]))
        predDict = dict()
        featureImpList = list()
        for i, combTest in enumerate(skuCombsTesting):
            print ("")
            print ("==================================================")
            print ("___on training SKU portion %s " % str(splitPor))
            print ("___on Test SKU pair %s " % str(combTest))
            print ("___on Test SKU id %d of %d " % (i+1,len(skuCombsTesting)))
            trainIDs_both, testIDs_both, trainSets_both, testSets_both, sku_test_set_both = _normalSplitPhoneSKU(trainingStage,combTest,trainingStageMarker)
            modelName_both = "normalSplit_" + str(sku_test_set_both) +  "_"+ trainingStageMarker["gender"]+"_"+ str(i)
            lgb_splitBoth = lgbBoosterEearlyStopping(trainSets_both, testSets_both, serilierModel, modelName_both)

            pred_y = _singleModelCheck(lgb_splitBoth, testingDS, "SplitNormal_round_" +"_"+trainingStageMarker["gender"]+ str(i))
            predDict[str(combTest)] = pred_y
            featureImpList.append(lgb_splitBoth.feature_importance())

        featureMeanIms = np.mean(featureImpList,axis=0)
        corMean,corStd = _iterAna(testingDS[1],predDict,"SKUcombsANA_"+str(len(skuCombsTesting[0]))+"_"+trainingStageMarker["gender"])
        serilizerBuffer.saveJoblib(predDict,"SKUcombs_"+str(len(skuCombsTesting[0]))+"_"+trainingStageMarker["gender"])
        serilizerBuffer.saveJoblib(featureMeanIms,"SKUcombsImp_"+str(len(skuCombsTesting[0]))+"_"+trainingStageMarker["gender"])
        serilizerBuffer.saveJoblib([corMean,corStd],"SKUcombsANA_"+str(len(skuCombsTesting[0]))+"_"+trainingStageMarker["gender"])
        corAnaDict[len(skuCombsTesting[0])] = [corMean,corStd]
        print (" Summary of TEST SKU numer %d " % len(skuCombsTesting))
        print ("Cor mean %f " % corMean)
        print ("Cor std  %f " % corStd)
        print ("entire cor ana is:")
        for k,v in corAnaDict.items():
            print (k)
            print (v)

def _splitPolicyCompare(gender):
    '''
        resDict = {"v1":Response_v1,"v81":Response_v8_1,"v82":Response_v8_2}
        footDict = {"foot01":FootClassical_01,"foot04":FootClassical_04,"foot05":FootClassical_05}
        shoeDict = {"shoeClassical":ShoesClassical,"shoeCalibrate":ShoesCalibrate,"shoeAll04":ShoesClassicalAll_04,"shoeAll05":ShoesClassicalAll_05}
        '''

    ###----- compile training data ---
    footVersion = "foot01"
    shoeVersion = "shoeAll06"
    newShoeFeatureSeg = 5

    stageMarker_v1 = {
        "gender": gender,
        "footVersion": footVersion,
        "shoeVersion": shoeVersion,
        "shoeFeatureSeg": newShoeFeatureSeg,
        "resVersion": "v1",
        "phoneProportion": 0.8,
        "skuProportion": 0.75,
    }
    footFeatures = classicalFoot_v1["contFeatures"] + classicalFoot_v1["orderedDisc"]
    # footFeatures = classicalFoot_offline
    # shoeFeatures0 = classicalShoe["contFeatures"]+ classicalShoe["orderedDisc"] + classicalShoe["unorderedDisc"]
    shoeFeatures = ["gender"] + classicalShoe["contFeatures"]
    # stageShoeFeatureEng = stageShoeFeatures(shoeFeatures,stageMarker_v1)

    stage1_v1 = stageOne(gender, footFeatures, shoeFeatures, stageMarker_v1)
    '''stage1 ::  [[phone,sku,size,....]...] '''

    stageMarker_v81 = stageMarker_v1
    stageMarker_v81["resVersion"] = "v81"
    stage1_v81 = stageOne(gender, footFeatures, shoeFeatures, stageMarker_v81)
    v81_ids, v81_datset = _normalTrans(stage1_v81, stageMarker_v81)

    stageMarker_v82 = stageMarker_v1
    stageMarker_v82["resVersion"] = "v82"
    stage1_v82 = stageOne(gender, footFeatures, shoeFeatures, stageMarker_v82)
    v82_ids, v82_datset = _normalTrans(stage1_v82, stageMarker_v82)

    exePath = realpath(__file__)
    serilierModel = Serializer(exePath,"models")

    skuCombs = _getSKUcombs(stage1_v1,stageMarker_v1["skuProportion"])
    # for i in range(40):
    for i,comb in enumerate(skuCombs):
        # split SKU and Phone
        print ("")
        print ("__<<____<<____ split phone and sku ___>>___>>>____")
        trainIDs_both,testIDs_both, trainSets_both, testSets_both,sku_test_set_both = _normalSplitPhoneSKU(stage1_v1,comb,stageMarker_v1)
        modelName_both = "normalSplit_" + str(sku_test_set_both) +  "_"+ stageMarker_v1["gender"]+"_"+ str(i)
        lgb_splitBoth = lgbBoosterEearlyStopping(trainSets_both,testSets_both,serilierModel,modelName_both)
        ## splitSKUonly
        print ("")
        print ("__<<____<<____ split sku only ___>>___>>>____")
        trainIDs_SKU,testIDs_SKU, trainSets_SKU, testSets_SKU,sku_test_set_SKU = _normalSplitSKU(stage1_v1,comb,stageMarker_v1)
        modelName_SKU = "normalSplitSKU_" + str(sku_test_set_SKU) +  "_"+ stageMarker_v1["gender"]+"_"+ str(i)
        lgb_splitSKU = lgbBoosterEearlyStopping(trainSets_SKU,testSets_SKU,serilierModel,modelName_SKU)
        ## splitPhoneOnly
        print ("")
        print ("__<<____<<____ split phone only___>>___>>>____")
        trainIDs_Phone,testIDs_Phone, trainSets_Phone, testSets_Phone = _normalSplitPhone(stage1_v1,stageMarker_v1)
        modelName_Phone= "normalSplitPhone_" + str(comb)+"_"+stageMarker_v1["gender"]+"_"+str(i)
        lgb_splitPhone = lgbBoosterEearlyStopping(trainSets_Phone,testSets_Phone,serilierModel,modelName_Phone)

        print ("")
        print ("<+++++++ Check on V81+++++++++")
        _singleModelCheck(lgb_splitBoth,v81_datset,"SplitBoth_round_"+str(i))
        _singleModelCheck(lgb_splitSKU,v81_datset,"SplitSKU_round_"+str(i))
        _singleModelCheck(lgb_splitPhone,v81_datset,"SplitPhone_round_"+str(i))
        print ("")
        print ("<+++++++ Check on V82++++++++")
        _singleModelCheck(lgb_splitBoth, v82_datset,"SplitBoth_round_"+str(i))
        _singleModelCheck(lgb_splitSKU,  v82_datset,"SplitSKU_round_"+str(i))
        _singleModelCheck(lgb_splitPhone,v82_datset,"SplitPhone_round_"+str(i))
        print ("=====+++===+++  ROUND %d finished +++====++++====" % i)

def assembleStages(gender,footFeatures,shoeFeatures):
    '''
    resDict = {"v1":Response_v1,"v81":Response_v8_1,"v82":Response_v8_2}
    footDict = {"foot01":FootClassical_01,"foot04":FootClassical_04,"foot05":FootClassical_05}
    shoeDict = {"shoeClassical":ShoesClassical,"shoeCalibrate":ShoesCalibrate,"shoeAll04":ShoesClassicalAll_04,"shoeAll05":ShoesClassicalAll_05}
    '''

    ###----- compile training data ---
    footVersion = "foot01"
    shoeVersion = "shoeAll06"
    newShoeFeatureSeg = 5

    stageMarker_v1 = {
       "gender":gender,
       "footVersion":footVersion,
       "shoeVersion":shoeVersion,
       "shoeFeatureSeg":newShoeFeatureSeg,
        "resVersion":"v1",
        "phoneProportion":0.8,
        "skuProportion":0.75,
    }

    stage1_v1 = stageOne(gender,footFeatures,shoeFeatures,stageMarker_v1)
    '''stage1 ::  [[phone,sku,size,....]...] '''

    stageMarker_v81 = stageMarker_v1
    stageMarker_v81["resVersion"] = "v81"
    stage1_v81 = stageOne(gender,footFeatures,shoeFeatures,stageMarker_v81)
    v81_ids, v81_datset = _normalTrans(stage1_v81,stageMarker_v81)

    # splitRange = [0.75,0.5,0.25,0.10]
    splitRange = [0.80,0.70,0.60,0.50,0.40,0.30,0.20,0.10]
    iterOverSKUcomb(stage1_v1,v81_datset,stageMarker_v1,splitRange)


def _skuCombANA(gender,featureList,sfLength):
    exePath = realpath(__file__)
    modelFolder= (dirname(exePath)) + "/tmp"
    serilizerBuffer = Serializer(exePath, "tmp")
    print (modelFolder)
    anaDict = dict()
    ImportanceDict = dict()
    pDictList = list()

    for root, dirs, mfiles in os.walk(modelFolder,topdown=False):
        for f in mfiles:
            g = f.split("_")[-1]
            if g != gender: continue
            testSKUnum = f.split("_")[-2]
            taskName = f.split("_")[0]
            obj,t = serilizerBuffer.loadJoblib(f)
            if taskName == "SKUcombs":
                pDictList.append(obj)
            if taskName == "SKUcombsANA":
                anaDict[testSKUnum] = obj
            if taskName == "SKUcombsImp":
                ImportanceDict[testSKUnum] = obj
            print (f)
    print (featureList)
    print (len(featureList))

    for k,im in ImportanceDict.items():
        print ("")
        print (k)
        print (anaDict.get(k))
        print (im[-sfLength:])
        # print (im[:-sfLength])
        print (len(im[-sfLength:]))
        print (sfLength)

    workbook = xlsxwriter.Workbook("ana_"+str(gender)+'.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    worksheet.write_row(0,0,["test_sku_num"]+featureList[-sfLength:])
    for k , im in ImportanceDict.items():
        row +=1
        tl = list(im[-sfLength:])
        data = list(map(lambda x:float(x) ,[int(k)]+tl))
        worksheet.write_row(row,0,data)
    workbook.close()
    return anaDict,ImportanceDict,pDictList

def _modelANA(gender,footFeatures,shoeFeatures):
    footVersion = "foot01"
    shoeVersion = "shoeAll06"
    newShoeFeatureSeg = 5

    stageMarker_v1 = {
       "gender":gender,
       "footVersion":footVersion,
       "shoeVersion":shoeVersion,
       "shoeFeatureSeg":newShoeFeatureSeg,
        "resVersion":"v1",
        "phoneProportion":0.8,
        "skuProportion":0.75,
    }

    stageMarker_v81 = stageMarker_v1
    stageMarker_v81["resVersion"] = "v81"
    stage1_v81 = stageOne(gender,footFeatures,shoeFeatures,stageMarker_v81)
    v81_ids, v81_datset = _normalTrans(stage1_v81,stageMarker_v81)
    print ("")

    check_dict_v81 = _modelscheckDict(v81_datset,stageMarker_v81,modelname="normalSplit")

    # for ind,v in check_dict_v81.items():
    #     print (ind)

    reportArray = list()
    verifyArray = list()
    reportRagne = range(len(check_dict_v81.keys()))
    for r in reportRagne:
        for k,v in check_dict_v81.items():
            rindex = int(k)
            if rindex != r:
                continue
            print (k)
            reportArray.append(v)
            ttlist = list(map(_ttF,list(zip(v81_datset[1],v))))
            # ttlist = list(map(_ttFSmaller,list(zip(v81_datset[1],v))))
            verifyArray.append(ttlist)


    verificationSum = np.sum(verifyArray, axis=0)
    print(len(verificationSum))
    plt.hist(verificationSum, bins=range(30))
    print(len(verificationSum))

    prefectList = [v81_ids[i] for i, p in enumerate(verificationSum) if p == len(check_dict_v81.keys())]
    loserList = [v81_ids[i] for i, p in enumerate(verificationSum) if p == 0]
    print(len(prefectList))
    print(len(loserList))
    plt.show()
    print("perfect phone")
    perfectPhone = (list(set([p[0] for p in prefectList])))
    print(perfectPhone)
    print("perfect sku")
    perfectSKU = (list(set([p[1] for p in prefectList])))
    print(perfectSKU)
    print("loser phone")
    loserPhone = (list(set([p[0] for p in loserList])))
    print(loserPhone)
    print("loser sku")
    loserSKU = (list(set([p[1] for p in loserList])))
    print(loserSKU)
    totalSKU = (list(set([ids[1] for ids in v81_ids])))
    print ("total SKU")
    print (totalSKU)
    totalPhone = (list(set([ids[0] for ids in v81_ids])))

    joinPhone = [p for p in perfectPhone if p in loserPhone]
    print("")
    print ("joinPhone")
    print(joinPhone)

    joinSKU = [p for p in perfectSKU if p in loserSKU]
    print ("join SKU")
    print(joinSKU)

    print ("sku total : %d" %len(totalSKU))
    print ("sku perfect: %d" %len(perfectSKU))
    print ("sku loser:   %d" %len(loserSKU))
    print ("sku join :   %d" %len(joinSKU))
    print ("")
    print ("phone total : %d"  %len(totalPhone))
    print ("phone perfect: %d" %len(perfectPhone))
    print ("phone loser:   %d" %len(loserPhone))
    print ("phone join :   %d" %len(joinPhone))

    purePerfectPhone = [p for p in perfectPhone if p not in loserPhone]
    pureLoserPhon = [p for p in loserPhone if p not in perfectPhone]
    print ("pure perfect phone %d " %len(purePerfectPhone))
    print ("pure loser phone %d " % len(pureLoserPhon))

    sDict = dict()
    for i,v in enumerate(verificationSum):
        sku = v81_ids[i][1]
        if v != 0 :continue
        tmp = sDict.get(sku,0)
        tmp+=1
        sDict[sku] = tmp
    print (sDict)

    import operator
    sorted_x = sorted(sDict.items(), key=operator.itemgetter(1))
    print (sorted_x)

    pDict = dict()
    for i,v in enumerate(verificationSum):
        phone = v81_ids[i][0]
        if v != 0 :continue
        if phone in perfectPhone: continue
        tmp = pDict.get(phone,0)
        tmp+=1
        pDict[phone] = tmp
    sorted_p = sorted(pDict.items(), key=operator.itemgetter(1))
    print (sorted_p)
    print (len(sorted_p))

    sDict = dict()
    for i,v in enumerate(verificationSum):
        sku = v81_ids[i][1]
        if v != 0 :continue
        tmp = sDict.get(sku,0)
        tmp+=1
        sDict[sku] = tmp
    sorted_s = sorted(sDict.items(), key=operator.itemgetter(1))
    print (sorted_s)
    print (len(sorted_s))

    # for i in v81_ids:
    #     phone = i[0]
    #     if phone in pureLoserPhon:
    #         print (i)

if __name__ == "__main__":
    footFeatures = classicalFoot_v1["contFeatures"]+classicalFoot_v1["orderedDisc"]
    # footFeatures = classicalFoot_offline
    # shoeFeatures0 = classicalShoe["contFeatures"]+ classicalShoe["orderedDisc"] + classicalShoe["unorderedDisc"]
    shoeFeatures = ["gender"]+classicalShoe["contFeatures"]
    # stageShoeFeatureEng = stageShoeFeatures(shoeFeatures,stageMarker_v1)

    ### ======== general sku combination search ==========
    # genders = ["1"]
    # for g in genders:
    #     assembleStages(g,footFeatures=footFeatures,shoeFeatures=shoeFeatures)

    ### ==== sku combination ana ======
    # featureList = footFeatures+footFeatures+shoeFeatures[1:]
    # haha = _skuCombANA("1",featureList,len(shoeFeatures[1:]))
    # print("w")

    ## ================= split policy search
    # _splitPolicyCompare("1")
    # _loadBoosters()

    ##==========
    _modelANA("1",footFeatures=footFeatures,shoeFeatures=shoeFeatures)

