from itertools import chain
from os.path import dirname, realpath

from wh.basicProcessor.v1process import *
from wh.basicProcessor.v8process import *
from wh.basicProcessor.featureInfor import *
from wh.basicProcessor.FootProcess import *
from wh.basicProcessor.ShoeProcess import *
from wh.basicProcessor.serializer import *

from wh.lastMonitor.simpleTraining import *

FEMALE_SIZE_RANGE_FILE_NAME = "3_last_female_sizeRange.pkl"
MALE_SIZE_RANGE_FILE_NAME = "3_last_male_sizeRange.pkl"

Response_v1 = "/home/MachineLearning/Datasets/DataInspector/Response/v1response.csv"
Response_v8_1 = "/home/MachineLearning/Datasets/DataInspector/Response/v8response_v1.csv"
Response_v8_2 = "/home/MachineLearning/Datasets/DataInspector/Response/v8response_v2.csv"

ShoesClassical = "/home/MachineLearning/Datasets/DataInspector/Shoes/data_out_sku.csv"
ShoesCalibrate = "/home/MachineLearning/Datasets/DataInspector/Shoes/data_out_sku_calibrate.csv"
ShoesClassicalAll_04 = "/home/MachineLearning/Datasets/DataInspector/Shoes/data_out_sku_201904.csv"
ShoesClassicalAll_05 = "/home/MachineLearning/Datasets/DataInspector/Shoes/data_out_sku_201905.csv"

FootClassical_01 = "/home/MachineLearning/Datasets/DataInspector/Foot/data_out_phone.csv"
FootClassical_04 = "/home/MachineLearning/Datasets/DataInspector/Foot/data_out_phone_201904.csv"
FootClassical_05 = "/home/MachineLearning/Datasets/DataInspector/Foot/data_out_phone_201905.csv"

resDict = {"v1":Response_v1,"v81":Response_v8_1,"v82":Response_v8_2}
shoeDict = {"shoeClassical":ShoesClassical,"shoeCalibrate":ShoesCalibrate,"shoeAll04":ShoesClassicalAll_04,"shoeAll05":ShoesClassicalAll_05}
footDict = {"foot01":FootClassical_01,"foot04":FootClassical_04,"foot05":FootClassical_05}

def inspectCompletness():
    def _listJoin(list1,list2):
        oklist = [l for l in list1 if l in list2]
        nooklist = [l for l in list1 if l not in oklist]
        return oklist,nooklist

    records = getSizeV1(Response_v1)
    shoeSetv1 =  list(set([s[1] for s in records]))
    phoneSetv1 = list(set([s[0] for s in records]))

    records = getSizeV8(Response_v8_1)
    shoeSetv81 =  list(set([s[1] for s in records]))
    phoneSetv81 = list(set([s[0] for s in records]))

    records = getSizeV8(Response_v8_2)
    shoeSetv82 =  list(set([s[1] for s in records]))
    phoneSetv82 = list(set([s[0] for s in records]))

    shoeset1 = list(set(list(chain.from_iterable(getShoeFeatures(ShoesClassical,["sku_no"])))))
    shoeset2 = list(set(list(chain.from_iterable(getShoeFeatures(ShoesCalibrate,["sku_no"])))))
    shoeset3 = list(set(list(chain.from_iterable(getShoeFeatures(ShoesClassicalAll_04,["sku_no"])))))
    shoeset4 = list(set(list(chain.from_iterable(getShoeFeatures(ShoesClassicalAll_05,["sku_no"])))))

    shoeSource = [shoeset1,shoeset2,shoeset3,shoeset4]
    for s in shoeSource:
        print (len(s))

    for sSet in [shoeSetv1,shoeSetv81,shoeSetv82]:
        for s in shoeSource:
            okl,nokl = _listJoin(sSet,s)
            print ("looking for "+str(len(sSet))+" in "+str(len(s)))
            print (okl)
            print (nokl)
        print ("")

    print ("all possible sku in v1, v8.1 v8.2 list:")
    print (list(set((shoeSetv1+shoeSetv81+shoeSetv82))))

    print ("")
    print ("Looking for phone")
    phoneset1 = list(set(list(chain.from_iterable(getFooFeatures(FootClassical_01, ["phone"])))))
    phoneset2 = list(set(list(chain.from_iterable(getFooFeatures(FootClassical_04, ["phone"])))))
    phoneset3 = list(set(list(chain.from_iterable(getFooFeatures(FootClassical_05, ["phone"])))))

    phoneSource = [phoneset1, phoneset2, phoneset3]
    for s in phoneSource:
        print(len(s))

    for pSet in [phoneSetv1,phoneSetv81,phoneSetv82]:
        for p in phoneSource:
            okl, nokl = _listJoin(pSet, p)
            print("looking for " + str(len(pSet)) + " in " + str(len(p)))
            print(len(okl))
            print(len(nokl))
        print("")


def testFunctions():
    records = getSizeV1(Response_v1)
    print (len(records))
    print (records[0])
    records = getSizeV8(Response_v8_1)
    print (len(records))
    print (records[0])
    records = getSizeV8(Response_v8_2)
    print (len(records))
    print (records[0])

    print (classicalFoot["contFeatures"])
    print (classicalFoot["orderedDisc"])
    print (classicalFoot["unorderedDisc"])

    print (classicalShoe["contFeatures"])
    print (classicalShoe["orderedDisc"])
    print (classicalShoe["unorderedDisc"])

    print (len(classicalFoot["contFeatures"]))
    print (len(classicalFoot_v1["contFeatures"]))

def retrieveDatasets(resVer,footVer,footFeatures,shoeVer,shoeFeatures,gender):
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

        if lfv is None or rfv is None or sfv is None:
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

def split4Monitor(stage1,proportion,gender):
    skuSet = list(set([r[1] for r in stage1]))
    phoneSet = list(set([r[0] for r in stage1]))

    if gender == "1":
        size_range = joblib.load(_get_meta_data(MALE_SIZE_RANGE_FILE_NAME))
    else:
        size_range = joblib.load(_get_meta_data(FEMALE_SIZE_RANGE_FILE_NAME))

    # print (size_range)
    # print (len(size_range))

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

def stageFour(stage2,stage3):
##---------------------- Stage 4 -------------------------------------------------###
    exePath = realpath(__file__)
    serilizer = Serializer(exePath,"buffer")
    compare(serilizer,stage2,stage3)

def stageThree(stage2,reTrain=False):
##---------------------- Stage 3 -------------------------------------------------###
    print ("")
    exePath = realpath(__file__)
    serilizer = Serializer(exePath,"buffer")
    stage3 = trainModel(stage2,serilizer,reTrain)
    return stage3


def stageTwo(stage1,gender):
##------------------------- Stage 2 ----------------------------------------------###
    stage2 = split4Monitor(stage1,0.8,gender)
    '''
    (sku,traingSet,validSet)
    trainSet , validSet ::[[phone,sku,size,....]]
    '''
    print ("")
    print ("stage two")
    print (len(stage2))
    print (len(stage2[0]))
    print (len(stage2[0][1]))
    print (len(stage2[0][2]))
    return stage2

def stageOne(gender,footFeatures,shoeFeatures):
##------------------Stage 1 ---------------------------------------------------###
    stage1 = retrieveDatasets("v1","foot01",footFeatures,"shoeClassical",shoeFeatures,gender)
    print ("")
    print ("stage one")
    print (stage1[0])
    print (stage1[1])
    print (len(stage1))
    return stage1

def assembleStages():
    '''
    resDict = {"v1":Response_v1,"v81":Response_v8_1,"v82":Response_v8_2}
    footDict = {"foot01":FootClassical_01,"foot04":FootClassical_04,"foot05":FootClassical_05}
    shoeDict = {"shoeClassical":ShoesClassical,"shoeCalibrate":ShoesCalibrate,"shoeAll04":ShoesClassicalAll_04,"shoeAll05":ShoesClassicalAll_05}
    '''
    # testFunctions()
    # inspectCompletness()

    gender = "1"
    footFeatures = classicalFoot_v1["contFeatures"]+classicalFoot_v1["orderedDisc"]
    # shoeFeatures0 = classicalShoe["contFeatures"]+ classicalShoe["orderedDisc"] + classicalShoe["unorderedDisc"]
    shoeFeatures = ["gender"]+classicalShoe["contFeatures"]

    ###----- compile training data ---
    stage1 = stageOne(gender,footFeatures,shoeFeatures)

    ### ----- split to desire sections ---
    stage2 = stageTwo(stage1,gender)

    ## ------ train models -------
    retrain = False
    stage3 = stageThree(stage2,retrain)
    #
    ## ----- analyse results ------
    stageFour(stage2,stage3)

    # inspectModels(stage2,footFeatures,shoeFeatures)

if __name__ == "__main__":
    assembleStages()

