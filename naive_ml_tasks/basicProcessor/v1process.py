import csv
Response_v1 = "/home/MachineLearning/Datasets/DataInspector/Response/v1response.csv"

def getSizeV1(filePath):

    records = list()
    with open(filePath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            comment = int(row[3].split(")")[0])
            if comment != 0:
                continue
            if not filterLogic(row[4:]):continue
            phone = row[0]
            sku = row[1]
            size = row[2].split("(")[-1]
            records.append([phone,sku,size])

    return records


def filterLogic(anwlist):
    def _singlePosition(overall,questions):
        worest_response = len([q for q in questions if q == 1 or q == 5])
        best_response = len([q for q in questions if q == 3])

        if overall == 1:  ## less than 10% 1 or 5 in each part to get a total 1.
            if worest_response / len(questions) > 0.19:
                return False
            else:
                return True
        if overall == 2:
            if worest_response / len(questions) > 0.39:  ## less than 30% 1 or 5 in each part to get a total 1.
                return False
            else:
                return True
        if overall == 3:  ## no worset case and over 75% best response will be false.
            if best_response / len(questions) > 0.70 and worest_response == 0:
                return False
            else:
                return True

    leftList = list()
    rightList = list()
    anumList = list()


    for i,anw in enumerate(anwlist):
        if i%2 == 0:
            anwNum = anw.split(":")[0]
            anumList.append(int(anwNum))
            leftList.append(int(anw.split("(")[-1]))
        else:
            rightList.append(int(anw.split(")")[0]))
    overallLeft = None
    overallright = None
    if len(anumList) > 13: return False

    for i,anum in enumerate(anumList):
        if anum == 12:
            overallLeft  = leftList[i]
            overallright = rightList[i]
            del leftList[i]
            del rightList[i]

    leftFeelsGood = _singlePosition(overallLeft,leftList)
    rightFeelsGood = _singlePosition(overallright,rightList)

    return (leftFeelsGood and rightFeelsGood)



if __name__ == "__main__":
    getSizeV1(Response_v1)