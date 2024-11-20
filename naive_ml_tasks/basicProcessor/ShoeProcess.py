import csv

ShoesClassical = "/home/MachineLearning/Datasets/DataInspector/Shoes/data_out_sku.csv"

def getShoeFeatures(filePath,feature_list,withKey = False):

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
                records.append([row[0],row[2]]+vlist)
                # records.append([row[0],row[2]]+[ir for ii,ir in enumerate(row) if ii in indexList])

    return records



if __name__ == "__main__":
    getShoeFeatures(ShoesClassical,["sku_no"],True)