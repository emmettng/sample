import itertools
from numpy.linalg import norm
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


from wh.basicProcessor.v1process import *
from wh.basicProcessor.v8process import *
from wh.basicProcessor.featureInfor import *
from wh.basicProcessor.FootProcess import *
from wh.basicProcessor.ShoeProcess import *
from wh.basicProcessor.serializer import *

from wh.booster.simpleBooster import *

FEMALE_SIZE_RANGE_FILE_NAME = "3_last_female_sizeRange.pkl"
MALE_SIZE_RANGE_FILE_NAME = "3_last_male_sizeRange.pkl"

Response_v1 = "/home/MachineLearning/Datasets/DataInspector/Response/v1response.csv"
Response_v8_1 = "/home/MachineLearning/Datasets/DataInspector/Response/v8response_v1.csv"
Response_v8_2 = "/home/MachineLearning/Datasets/DataInspector/Response/v8response_v2.csv"

ShoesClassical = "/home/MachineLearning/Datasets/DataInspector/Shoes/data_out_sku.csv"
ShoesCalibrate = "/home/MachineLearning/Datasets/DataInspector/Shoes/data_out_sku_calibrate.csv"
ShoesClassicalAll_04 = "/home/MachineLearning/Datasets/DataInspector/Shoes/data_out_sku_201904.csv"
ShoesClassicalAll_05 = "/home/MachineLearning/Datasets/DataInspector/Shoes/data_out_sku_201905.csv"
ShoesClassicalAll_06 = "/home/MachineLearning/Datasets/DataInspector/Shoes/data_out_sku_20190617.csv"

FootClassical_01 = "/home/MachineLearning/Datasets/DataInspector/Foot/data_out_phone.csv"
FootClassical_04 = "/home/MachineLearning/Datasets/DataInspector/Foot/data_out_phone_201904.csv"
FootClassical_05 = "/home/MachineLearning/Datasets/DataInspector/Foot/data_out_phone_201905.csv"

SIZE_RANGE= list(range(200,290,5))

resDict = {"v1":Response_v1,"v81":Response_v8_1,"v82":Response_v8_2}
shoeDict = {"shoeClassical":ShoesClassical,"shoeCalibrate":ShoesCalibrate,"shoeAll04":ShoesClassicalAll_04,"shoeAll05":ShoesClassicalAll_05,"shoeAll06":ShoesClassicalAll_06}
footDict = {"foot01":FootClassical_01,"foot04":FootClassical_04,"foot05":FootClassical_05}

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
    shoeV = shoeDict.get(shoeVer,None)
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

def _retrieveDatasets(resVer,footVer,footFeatures,shoeVer,shoeFeatures,engShoeFeatures,gender):
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
    resV = resDict.get(resVer,None)
    footV = footDict.get(footVer,None)
    shoeV = shoeDict.get(shoeVer,None)

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
        lfv = lfDict.get(phone,None)
        rfv = rfDict.get(phone,None)
        # sfv = sDict.get((sku,size),None)
        sfv = sDict.get(sku,None)
        snfv = engShoeFeatures.get(sku,None)

        if lfv is None or rfv is None or sfv is None or snfv is None:
            continue
        lfvNum = list(map(lambda x:float(x),lfv))
        rfvNum = list(map(lambda x:float(x),rfv))
        sfvNum = list(map(lambda x:float(x),sfv))

        stageOne.append(r+lfvNum+rfvNum+sfvNum+snfv)
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


def stageTwo(stage1,proportion):
    ##------------------------- Stage 2 ----------------------------------------------###
    stage2 = split4Monitor(stage1,proportion)
    '''
    (sku,traingSet,validSet)
    trainSet , validSet ::[[phone,sku,size,....]]
    '''
    print ("")
    print ("stage two")
    print ("relate num of sku: %d " % len(stage2))
    print ("Record dimension of each sku: %d " % len(stage2[0]))
    print ("number of training reocrds: %d "% len(stage2[0][1]))
    print ("number of validation records: %d " % len(stage2[0][2]))
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

def stageOne(gender,footFeatures,shoeFeatures,engShoeFeatures,stageMarker):
    ##------------------Stage 1 ---------------------------------------------------###
    def _decodeStage1(stage1):
        print ("Stage One length: %d" % len(stage1))
        print (stage1[0])
        print (len(stage1[0]))
        print (stage1[1])

    resVer = stageMarker.get("resVersion")
    footVer = stageMarker.get("footVersion")
    shoeVer = stageMarker.get("shoeVersion")

    stageName = list(map(lambda x:str(x),stageMarker.values()))
    tmpFileName = "_".join(["stageOne"]+stageName)
    print ("")
    exePath = realpath(__file__)
    serilizerBuffer = Serializer(exePath,"buffer")
    stage1, t = serilizerBuffer.loadJoblib(tmpFileName)
    if t is not None:
        print ("load stage1 %s from disk" % tmpFileName)
        _decodeStage1(stage1)
        print ("")
        return stage1
    stage1 = _retrieveDatasets(resVer,footVer,footFeatures,shoeVer,shoeFeatures,engShoeFeatures,gender)
    serilizerBuffer.saveJoblib(stage1,tmpFileName)
    print ("")
    print ("Process stage one from csv")
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

def normalSplit(stage1,stageMarker):
    skuSet = list(set([r[1] for r in stage1]))
    phoneSet = list(set([r[0] for r in stage1]))
    size_range = SIZE_RANGE

    stage2_train= list()
    stage2_test= list()

    phone_sample_num = round(0.75* len(phoneSet))
    phone_training_set = random.sample(phoneSet, phone_sample_num)
    # sku_sample_num = round(0.75* len(skuSet))
    # sku_training_set = random.sample(skuSet, sku_sample_num)
    print (len(stage1))
    print (len(skuSet))
    print (len(phoneSet))
    print (len(phone_training_set))
    # print (len(sku_training_set))
    trainIDs = list()
    testIDs = list()
    for record in stage1:

        inner_phone,inner_sku,inner_size = record[0],record[1],record[2]
        trainIDs.append([inner_phone,inner_sku,inner_size])
        size_code = size_range.index(int(inner_size))
        record[2] = size_code
        stage2_train.append(record)

        # size_code = size_range.index(int(inner_size))
        # record[2] = size_code
        # if inner_phone in phone_training_set:
        #     stage2_train.append(record)
        #
        # if inner_phone not in phone_training_set:
        #     testIDs.append([inner_phone,inner_sku,inner_size])
        #     stage2_test.append(record)

    train= _getDataset(stage2_train)
    # test= _getDataset(stage2_test)
    # singleTraining(train,test)
    # print (len(train[0]))
    # print (len(test[0]))
    return [trainIDs,[],train,[]]
##====================== normal split ===============

def _inspectModels(levelNmodels,x,y_true):
    reportDict = dict()
    for mN, l2m in enumerate(levelNmodels):
        print("Test on level 2 of model: %s " % mN)
        predList, accu, ttlist = _inspectLevelTwo(l2m, [x, y_true])
        reportDict[mN] = [predList, accu, ttlist]
    return reportDict


def assembleStages(gender):
    '''
    resDict = {"v1":Response_v1,"v81":Response_v8_1,"v82":Response_v8_2}
    footDict = {"foot01":FootClassical_01,"foot04":FootClassical_04,"foot05":FootClassical_05}
    shoeDict = {"shoeClassical":ShoesClassical,"shoeCalibrate":ShoesCalibrate,"shoeAll04":ShoesClassicalAll_04,"shoeAll05":ShoesClassicalAll_05}
    '''

    ###----- compile training data ---
    # gender = "1"
    resVersion = "v1"
    footVersion = "foot01"
    shoeVersion = "shoeClassical"
    newShoeFeatureSeg = 5

    exePath = realpath(__file__)
    serilizerBuffer = Serializer(exePath, "buffer")

    # stageMarker = [
    #     gender,
    #     resVersion,
    #     footVersion,
    #     shoeVersion,
    #     str(newShoeFeatureSeg)
    # ]
    stageMarker_v1 = {
       "gender":gender,
       "resVersion":resVersion,
       "footVersion":footVersion,
       "shoeVersion":shoeVersion,
       "shoeFeatureSeg":newShoeFeatureSeg,
        "phoneProportion":0.8,
        "skuProportion":0.75,
    }
    footFeatures = classicalFoot_v1["contFeatures"]+classicalFoot_v1["orderedDisc"]
    # shoeFeatures0 = classicalShoe["contFeatures"]+ classicalShoe["orderedDisc"] + classicalShoe["unorderedDisc"]
    shoeFeatures = ["gender"]+classicalShoe["contFeatures"]


    stageShoeFeatureEng = stageShoeFeatures(shoeFeatures,stageMarker_v1)
    stage1_v1 = stageOne(gender,footFeatures,shoeFeatures,stageShoeFeatureEng,stageMarker_v1)

    ### ----- split to desire sections ---
    phoneProportion = stageMarker_v1.get("phoneProportion")
    stage2_v1 = stageTwo(stage1_v1,phoneProportion)
    '''
    [(sku,traingSet,validSet)]
    trainSet , validSet ::[[phone,sku,size,....]]
    '''
    #
    #
    ### ----- combain sku together ---
    skuProportion = stageMarker_v1.get("skuProportion")
    recalculate = False
    stageName = list(map(lambda x: str(x), stageMarker_v1.values()))
    internalName_v1= "_".join(stageName+["internals.pkl"])
    obj,t = serilizerBuffer.loadJoblib(internalName_v1)
    print (internalName_v1)
    if recalculate or t is None:
        stage3_v1 = stageThree(stage2_v1,phoneProportion,skuProportion,stageMarker_v1,recalculate)
        '''
        {
            sku_pair: [ train_dataset, test_dataset, phone_leak2test ,phone_leak2train],
            ...
        }
        '''
        ## ------ training -------
        basicLevel, sndLevel = stackSKU(stage3_v1)
        x_internal,essemble_y,basicModels = stackingEntrance(basicLevel,sndLevel)
        print ("Save internal dataset to disk!")
        serilizerBuffer.saveJoblib([x_internal,essemble_y],internalName_v1)
    else:
        print ("")
        print ("load internal dataset from disk")
        x_internal,essemble_y = obj
        basicModels = _loadModels("levelOne")
        print ("debug")

    stageMarker_v81 = {
        "gender": gender,
        "resVersion": "v81",
        "footVersion": footVersion,
        "shoeVersion": shoeVersion,
        "shoeFeatureSeg": newShoeFeatureSeg,
        "phoneProportion": 0.8,
        "skuProportion": 0.75,
    }
    stageName_v81 = list(map(lambda x: str(x), stageMarker_v81.values()))
    internalName_v81= "_".join(stageName_v81+["internals.pkl"])
    [x_internal_v81,_ ],t_v81 = serilizerBuffer.loadJoblib(internalName_v81)


    stage1_v81 = stageOne(gender,footFeatures,shoeFeatures,stageShoeFeatureEng,stageMarker_v81)
    stage2_v81 = stageTwo(stage1_v81,phoneProportion)
    stage3_v81 = stageThree(stage2_v81,phoneProportion,skuProportion,stageMarker_v81,recalculate)
    basicLevel_v81, sndLevel_v81 = stackSKU(stage3_v81)

    ### ================= v82 =================###
    stageMarker_v82 = {
        "gender": gender,
        "resVersion": "v82",
        "footVersion": footVersion,
        "shoeVersion": shoeVersion,
        "shoeFeatureSeg": newShoeFeatureSeg,
        "phoneProportion": 0.8,
        "skuProportion": 0.75,
    }
    stage1_v82 = stageOne(gender,footFeatures,shoeFeatures,stageShoeFeatureEng,stageMarker_v82)
    stage2_v82_trainIDs, stage2_v82_testIDs, stage2_v82_train, stage2_v82_test= normalSplit(stage1_v82,stageMarker_v82)
    internal_v82_x = intermediaPred(basicModels,stage2_v82_train[0])
    internal_v82_y = stage2_v82_train[1]
    levelTwoModels= _loadModels("levelTwo")
    reportDict_level2_normal = _inspectModels(levelTwoModels,internal_v82_x,internal_v82_y)
    serilizerBuffer.saveJoblib([stage2_v82_trainIDs, reportDict_level2_normal],"report2Normal.pkl")
    ### ================= v82 =================###

    # essemble_x_v81 ,essemble_y_v81 =sndLevel_v81w
    # if t_v81 is None:
    #     print ("")
    #     print ("V81 essemble %d " % len(essemble_y_v81))
    #     x_internal_v81 = intermediaPred(basicModels,essemble_x_v81)
    #     serilizerBuffer.saveJoblib([x_internal_v81,essemble_y_v81],internalName_v81)
    #
    # train_x_v81=list()
    # train_y_v81=list()
    # test_x_v81=list()
    # test_y_v81=list()
    # for wl in basicLevel_v81:
    #     elementName, train_x,train_y,test_x,test_y = wl
    #     train_x_v81+=train_x
    #     train_y_v81+=train_y
    #     test_x_v81+=test_x
    #     test_y_v81+=test_y
    # train_v81 = [train_x_v81,train_y_v81]
    # test_v81= [test_x_v81,test_y_v81]
    #
    # exePath = realpath(__file__)
    # serializerV81 = Serializer(exePath, "models")
    # levelTwoModels = dict()
    # for i,wl in  enumerate(basicLevel_v81):
    #     elementName, train_x,train_y,test_x,test_y = wl
    #     modelName = "_".join(["levelTwo",str(i),elementName])
    #     bst,t_model_v81 = serializerV81.loadBooster(modelName)
    #     if t_model_v81 is None:
    #         print ("start training %s " % modelName)
    #         iter_train_x = intermediaPred(basicModels,train_x)
    #         iter_test_x = intermediaPred(basicModels,test_x)
    #         print ("training size %d " % len(iter_train_x))
    #         print ("testing size %d " % len(iter_test_x))
    #         bst = singleTraining([iter_train_x,train_y],[iter_test_x,test_y])
    #         serializerV81.saveBooster(bst,modelName,bst.best_iteration)
    #         levelTwoModels[modelName] = bst
    #     else:
    #         print ("loading model %s from disk" % modelName)
    #         levelTwoModels[modelName] = bst
    #
    # print (len(levelTwoModels.keys()))
    #
    # reportName = "reportDict.pkl"
    # reportDict, t_rep_v81 = serilizerBuffer.loadJoblib(reportName)
    # if t_rep_v81 is None:
    #     reportDict = dict()
    #     for mN,l2m in levelTwoModels.items():
    #         print ("Test on level 2 of model: %s " % mN)
    #         predList,accu,ttlist = _inspectLevelTwo(l2m,[x_internal_v81,essemble_y_v81])
    #         reportDict[mN] = [predList,accu,ttlist]
    #     serilizerBuffer.saveJoblib(reportDict,reportName)
    #
    # predictArray = list()
    # reportRagne = range(len(reportDict.keys()))
    # for r in reportRagne:
    #     for k,v in reportDict.items():
    #         rindex = int(k.split("_")[1])
    #         if rindex != r:
    #             continue
    #         print (k)
    #         predictArray.append(v[2])
    # predictSum = np.sum(predictArray,axis=0)
    # print (len(predictSum))
    # print (len(essemble_y_v81))
    # plt.hist(predictSum,bins=range(42))
    # print (len(predictSum))
    # print (len([e for e in predictSum if e == 40]))
    # print (len([e for e in predictSum if e == 39]))
    # print (len([e for e in predictSum if e == 0]))
    # print (len([e for e in predictSum if e == 1]))
    # print (len([e for e in predictSum if e > 20]))
    #
    # plt.show()


## ========= compute model correlation ===================###
    # reportArray = list()
    # reportRagne = range(len(reportDict.keys()))
    # for r in reportRagne:
    #     for k,v in reportDict.items():
    #         rindex = int(k.split("_")[1])
    #         if rindex != r:
    #             continue
    #         print (k)
    #         reportArray.append(v[0])
    #
    # print (len(reportArray[0]))
    # print ('de')
    #
    # correArray = np.zeros([len(reportArray),len(reportArray)])
    # for i , r in enumerate(reportArray):
    #     for innerI, innerR in enumerate(reportArray):
    #         correArray[i][innerI] = pearsonr(r,innerR)[0]
    #
    # print (correArray)
    # import xlsxwriter
    #
    # workbook = xlsxwriter.Workbook('arrays.xlsx')
    # worksheet = workbook.add_worksheet()
    # row = 0
    # for col, data in enumerate(correArray):
    #    worksheet.write_column(row,col,data)
    #
    # workbook.close()

## =================== feature importance ================ #
    # imList = []
    # for m in basicModels:
    #     print (m.feature_importance())
    #     print (len(m.feature_importance()))
    #     imList.append(m.feature_importance())
    # importans = np.mean(imList,axis=0)
    # print (importans)
    # print (len(importans))
    # wholeF = footFeatures+footFeatures+shoeFeatures
    # print (wholeF)
    #
    # for i, im in enumerate(importans):
    #     print (im)
    #     print (wholeF[i])
## =================== feature importance ================ #

    # singleTraining([x_internal,essemble_y],[x_internal_v81,essemble_y_v81])

    # bstLayerTwo = stackingLayerTwo(x_internal,essemble_y)
    ## ------ training -------
    # stackNum = 3
    # retrain = True
    # stage3 = stageTraining(stage3,stackNum,retrain)

    # ----- analyse results ------
    # stageFour(stage2,stage3)

    # inspectModels(stage2,footFeatures,shoeFeatures)

def anal():
    exePath = realpath(__file__)
    serilizerBuffer = Serializer(exePath, "buffer")
    tl,t = serilizerBuffer.loadJoblib("report2Normal.pkl")
    ids,reportDict = tl

    predictArray= list()
    verificationArray = list()
    reportRagne = range(len(reportDict.keys()))
    for r in reportRagne:
        for k,v in reportDict.items():
            rindex = k
            if rindex != r:
                continue
            predictArray.append(v[0])
            verificationArray.append(v[2])
    verificationSum= np.sum(verificationArray,axis=0)
    print (len(verificationSum))
    plt.hist(verificationSum,bins=range(42))
    print (len(verificationSum))
    print (len([e for e in verificationSum if e == 40]))
    print (len([e for e in verificationSum if e == 39]))
    print (len([e for e in verificationSum if e == 0]))
    print (len([e for e in verificationSum if e == 1]))
    print (len([e for e in verificationSum if e > 20]))

    prefectList = [ids[i] for i , p in enumerate(verificationSum) if p == 40]
    loserList = [ids[i] for i , p in   enumerate(verificationSum) if p == 0]
    print (len(prefectList))
    print (len(loserList))
    # plt.show()
    print ("perfect phone")
    perfectPhone = (list(set([p[0] for p in prefectList])))
    print (perfectPhone)
    print ("perfect sku")
    perfectSKU = (list(set([p[1] for p in prefectList])))
    print (perfectSKU )
    print ("loser phone")
    loserPhone = (list(set([p[0] for p in loserList])))
    print (loserPhone)
    print ("loser sku")
    loserSKU = (list(set([p[1] for p in loserList])))
    print (loserSKU)

    joinPhone = [p for p in perfectPhone if p in loserPhone]
    print ("")
    print (perfectPhone)
    print (joinPhone)

    joinSKU = [p for p in perfectSKU if p in loserSKU]
    print (perfectSKU)
    print (joinSKU)

    skuDict = _anaModelName("levelTwo")
    print (skuDict)

    [print(str(sku)+" : " + str(skuDict.get(sku))) for sku in joinSKU]
    print ("")
    [print(str(sku)+" : " + str(skuDict.get(sku))) for sku in loserSKU]

    for p in prefectList:
        print (p)

    print ("")
    for p in loserList:
        if p[0] not in joinPhone:continue
        print (p)

    print (len(ids))
    print (len(predictArray[0]))
    pDict = dict()
    for im,m in enumerate(predictArray):
        for ip, p in enumerate(m):
            tmpDict = pDict.get(ip,dict())
            tmpCnt = tmpDict.get(p,0)
            tmpCnt+=1
            tmpDict[p] = tmpCnt
            pDict[ip] = tmpDict

    for k,v in pDict.items():
        print (k)
        print (v)
        for ik,iv in v.items():
            print (SIZE_RANGE[ik])
            print (iv)
        print (ids[k])
        print ("")
if __name__ == "__main__":

    # genders = ["1"]
    # for g in genders:
    #     assembleStages(g)

    # anal()
    # resV = resDict.get("v82",None)
    # res = getSizeV8(resV)

    # shoeVer = "shoeClassical"
    # shoeVer = "shoeCalibrate"
    shoeVer = "shoeAll06"
    shoeFeatures = ["gender"]+classicalShoe["contFeatures"]
    shoeV = shoeDict.get(shoeVer,None)
    s4 = getShoeFeatures(shoeV,shoeFeatures,True)

    for s in ['230','240','250','260','270']:
        t1 = None
        t2 = None
        for r in s4:
            if r[0] == 'V91V2RB2C03CM8' and r[1] == s:
                t1 = r
            elif r[0] == 'V91V2102C03CM8' and r[1] == s:
                t2 = r
            else:
                continue
        # print (t1)
        # print (t2)
        print (list(map(lambda x:x[0]==x[1],list(zip(t1,t2)))))
