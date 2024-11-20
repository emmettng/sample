import csv

FootClassical_01 = "/home/MachineLearning/Datasets/DataInspector/Foot/data_out_phone.csv"

def getFooFeatures(filePath,feature_list,withKey = False):

    records = list()
    header = list()
    with open(filePath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            header = row
            break

    # indexList = [i for i, h in enumerate(header) if h in feature_list]
    indexList = [header.index(f) for f in feature_list if f in header]

    cnt = 0
    with open(filePath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if cnt == 0:
                cnt +=1
                continue
            vlist = [row[ind] for ind in indexList]
            if not withKey:
                records.append(vlist)
                # records.append([ir for ii,ir in enumerate(row) if ii in indexList])
            else:
                records.append([row[1]]+vlist)
                # records.append([row[1]]+[ir for ii,ir in enumerate(row) if ii in indexList])

    return records



if __name__ == "__main__":
    getFooFeatures(FootClassical_01,["phone"])
