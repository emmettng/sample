import pandas as pd
import time
import os
import os.path
import copy
import itertools

from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler,MinMaxScaler,MaxAbsScaler

fileName = './dataSource/diabetic_data.csv'
MODELPACKAGE = "./modelPck_"


KEYS = ['patient_nbr']
DROP_FEATURES = ['encounter_id','patient_nbr','weight','payer_code','medical_specialty',]
LABEL_FEATURE = ['readmitted']
LABEL_POSITIVE = '<30'

'''
Return path of working directory. For each combination of different score metric and scaler, a folder will be
created to contains all training results: model, encoder, exploration image, meta data, etc.

type:
                ()
                -> String
'''
def getModelPath():
    return MODELPACKAGE

'''
Save all kinds of different python object: model, metadata, etc.
'''
def _Save2pkl(obj,path):
    joblib.dump(obj,path)
    print ("%s has been serialized in path %s" % (str(type(obj)),path,))
def Save2pkl(obj,path):
    joblib.dump(obj,path)
    print ("%s has been serialized in path %s" % (str(type(obj)),path,))

'''
Drop columns with corresponding give names. Auxiliary function in this package.

type:           DataFrame           input pandas DataFrame
                -> [String]         column name list
                -> DataFrame        pandas DataFrame after processing.
'''
def _DropDFColumns(df, columns):
    assert type(df) == pd.core.frame.DataFrame, "df should be dataframe datatype"
    [df.__delitem__(col) for col in columns]
    return df

'''
Convert discrete string value into int. In this case (Task 2 and 3) all features are discrete non-ordinal features,
except 'age' but can be ignore this minor difference. This is a modification from an existing encoding function, so a little cumbersome.

type:           DataFrame -> DataFrame
'''
def _FeatureLabelEncoding(df,serialization=True):
    enc = LabelEncoder()
    encDict = dict()
    for i, col in enumerate(df.columns.values):
        if df[col].dtypes == 'object':
            data = df[col]
            enc.fit(data.values)
            df[col] = enc.transform(df[col])
            tmpEnc = copy.deepcopy(enc)
            encDict[i]=tmpEnc
    if serialization:
        _Save2pkl(encDict,MODELPACKAGE+"LabelEncoder.pkl")
    return df

'''
Apply one-hot encoding scheme to all features in this case.
By default, all original columns will be removed after encoding, and the encoded DataFrame will be save to local disk.

type:           DataFrame              DataFrame to be encoded
                -> [String]            columns need to be encoded
                -> DataFrame           encoded DataFrame
'''
def _FeatureOneHotEncoding(df,columns,removeOrigin=True,serialization=True):
    enc = OneHotEncoder(sparse=False)
    X_ohenc = df
    encDict = dict()
    for i,col in enumerate(columns):                    # fit encoder with target column
        data = df[[col]]
        enc.fit(data)                                   # transform target column and save the encoder
        tmp = enc.transform(df[[col]])
        tmpEnc = copy.deepcopy(enc)
        encDict[i] = tmpEnc                             # rebuild a dataFrame with new column name
        tmp = pd.DataFrame(tmp, columns=[(col + "_" + str(i)) for i in data[col]
                            .value_counts().index])     # Setting the index values similar to the df.
        tmp = tmp.set_index(df.index.values)            # Append new encoded DataFrame to existing DataFrame.
        X_ohenc = pd.concat([X_ohenc, tmp], axis=1)     # remove origin column if required.
    if removeOrigin:
        X_ohenc= _DropDFColumns(X_ohenc,columns)
    if serialization:
        _Save2pkl(encDict,MODELPACKAGE+"OneHotEncoder.pkl")         # Save encoded DataFrame to disk.
    return X_ohenc

'''
Remove rows with same feature value, keep either the first one or the last one.
Usually, duplicated personal id implies duplicate data that could lead to the fake increase of the training accuracy.

type:       DataFrame
            -> [String]
            -> DataFrame
'''
def _DropDuplication(df,features,keep_first=True):
    if keep_first:
        dDF= df.drop_duplicates(features)
    else:
        dDF = df.drop_duplicates(features,keep='last')
    return dDF

'''
Source data --> LabelEncoding --> OneHotEncoder --> Save to disk.
If the above process had been finished, then simply deserialize to get encoded X and y.
Otherwise, create corresponding folder and return folder path.

type:           String
                -> Either (String,String) (DataFrame,DataFrame)
'''
def _save2diskPreparation(v_tag):
    global MODELPACKAGE                     ## May GOD bless this thing
    folderPath = MODELPACKAGE+v_tag
    Xpath = folderPath+"/"+"X.pkl"
    ypath = folderPath+"/"+"y.pkl"
    MODELPACKAGE = folderPath+"/"
    if not os.path.exists(folderPath):
        os.mkdir(folderPath)
        return (False,(Xpath,ypath))
    if os.path.isfile(Xpath) and os.access(Xpath,os.R_OK) and os.path.isfile(ypath) and os.access(ypath,os.R_OK):
        print ("Loading X and y for training from pkl")
        X = joblib.load(Xpath)
        y = joblib.load(ypath)
        return (True,(X,y))
    return (False,(Xpath,ypath))

'''
Encode label to be either 0 or 1. Controlled by global variable 'LABEL_POSITIVE'.

type:           DataFrame -> DataFrame
'''
def LabelEncoding(y_df):
    y_df[y_df['readmitted'] == LABEL_POSITIVE] = 1
    y_df[y_df['readmitted'] != 1] = 0
    return y_df

'''
when the model is being used in real production environment, all input data need to be processed the same way as training data, before feeding into the model.
This function is being used to get input data for simulating real world use case.

type:           DataFrame
                -> DataFrame
'''
def featurePreprocess(featureDF):
    df = _DropDFColumns(featureDF,DROP_FEATURES)
    df = _DropDFColumns(df,LABEL_FEATURE)
    return df

'''
when the model is being used in real production environment, all input data need to be processed the same way as training data, before feeding into the model.
This function is being used to get input data for simulating real world use case.

type:           Encoder
                -> (Numpy array -> Numpy array)
'''
def featureCoding(lenc):
    def simpleCurry(record):
        codedData = list()
        for i, v in enumerate(record):
            if i not in lenc.keys():
                codedData.append(v)
                continue
            enc = lenc.get(i)
            newv = enc.transform([[v]])
            codedData.append(newv[0])
        return codedData
    return simpleCurry

'''
type:       DataFrame
            -> DataFrame
            -> ([String]
            -> [(DataFrame, DataFrame)]
'''
def contextSplit(X_df,y_df,features):
    cc = list()
    for c in features:
        cc.append(list(map(lambda x: {c: x}, list(set(X_df[c].values)))))
    a = list(itertools.product(*cc))
    tlist = list()
    for td in a:
        d = dict()
        dfbool = None
        [d.update(id) for id in list(td)]
        for k, v in d.items():
            if dfbool is None:
                dfbool = X_df[k] == v
                continue
            dfbool &= X_df[k] == v
        if (not any(dfbool.values)): continue
        X = X_df[dfbool]
        y = y_df[dfbool]
        tlist.append((X, y))
    return tlist

'''
A light version of 'PolynomialFeatures'.

type:       DataFrame
            -> [String]
            -> DataFrame
'''
def pairwiseFeatures(df,columns):
    for i,c in enumerate(columns):
        for r in columns[i:]:
            if r==c: continue
            df[str(c)+"_"+str(r)] = df[c]*df[r]
    return df

'''
Scaling encoded feature with different scaler before feeding into training process.

type:           DataFrame
                -> Numpy Array
'''
def featureScaling(X,s ='MinMax'):
    if s == 'Centering':
        scaler = StandardScaler()
    if s == 'MinMax':
        scaler = MinMaxScaler()
    if s == 'MaxAbs':
        scaler = MaxAbsScaler()
    X = scaler.fit_transform(X)
    _Save2pkl(scaler,MODELPACKAGE+str(s)+"Scaler.pkl")
    return X


'''
Remove features with low variance, in this case the  feature being removed can be identified by cardinality.

type:           DataFrame
                --> DataFrame
'''
def _removeNoVarianceFeatures(df):
    cardi = list(df.apply(pd.Series.nunique))
    cols = df.columns.values
    for i, v in enumerate(cardi):
        if v == 1:
            df = _DropDFColumns(df,[cols[i]])
        continue
    return df


'''
Compute Cardinality of all columns in give DataFrame, and drop features of cardinality 1.

type:           DataFrame
                -> [(Int, String)]
'''
def _computeFeaturecardinality(df):
    cardi = list(df.apply(pd.Series.nunique))
    cols = df.columns.values
    return dict(zip(cardi,cols))

'''
1. Being used to provide prediction base line.
2. Features that tightly related to some other feature could be ignored in the feature selection process,
such as when using 'l1' penalty together with logistic regression. However, the ignored feature might be able to
provide more explainability.

type:           DataFrame
                -> DataFrame
'''
def computeCorr(df):
    nonZeroDF= df.loc[:,(df != 0).any(axis=0)]
    corr = (nonZeroDF.corr())
    return corr
'''
Decode values of each feature to get literal description.

type:           [Encoder]           encoders being used
                -> [String]         column names need to be decoded
                -> [String]         result
'''
def decodeFeatures(encs,columns):
    # TODO
    return ()


'''
Mapping several values of a feature to a single value, reduce the potential model complexity and increase the explainability(Maybe).

type:           DataFrame
                -> [(Feature,([Feature_Value],Feature_value))]
                -> DataFrame
'''
def featureRecognize(df,featureMapping):
    # TODO
    return ()

'''

Compose all auxiliary functions in this file to provide basic data pre-processing functionality.

type:           [String]     Features that will not take part in the following training process.
                -> (DataFrame, DataFrame)    X for training and testing, y for training and testing
'''
def naivePreprocess(v_tag,drop_feature_list=DROP_FEATURES):
    flag,tup = _save2diskPreparation(v_tag)
    if flag:
        return tup
    else:
        Xpath, ypath = tup
        print ("Convert and encoding X and y for training from CSV file")
        print ("Please wait...")
        df = pd.read_csv(fileName)
        df = _DropDuplication(df,KEYS)                           # drop duplicated patient record
        newdf = _DropDFColumns(df,drop_feature_list)        # drop invalid features
        y = newdf[LABEL_FEATURE]                            # get y
        Xdf = _DropDFColumns(newdf,LABEL_FEATURE)           # get X
        Xdf = _FeatureLabelEncoding(Xdf)                    # apply LabelEncoder to X
        X = _FeatureOneHotEncoding(Xdf,Xdf.columns.values)  # apply OneHotEncoder to X
        _Save2pkl(X,Xpath)
        _Save2pkl(X.columns.values,MODELPACKAGE+"features.pkl")  # store feature name of each dimension
        print ("Please ignore next WARNING messages!")
        time.sleep(3)
        y = LabelEncoding(y)
        print ("Please ignore above WARNING messages!")
        time.sleep(3)
        _Save2pkl(y,ypath)
        return X, y

if __name__ =="__main__":
    print ("naivePreprocess is the composition of all encoding and filtering/mapping functions")

