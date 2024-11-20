import csv
Response_v8_1 = "/home/MachineLearning/Datasets/DataInspector/Response/v8response_v1.csv"
Response_v8_2 = "/home/MachineLearning/Datasets/DataInspector/Response/v8response_v2.csv"

def getSizeV8(filePath):

    records = list()
    with open(filePath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) == 0 :continue
            phone = row[0]
            sku = row[1]
            size = row[2]
            filterLogic(row[3:])

            if not filterLogic(row[3:]):continue
            records.append([phone,sku,size])

    return records


def filterLogic(anwlist):
    def _singlePosition(overall,questions):
        if overall == 1:
            return True
        else:
            return False

        # if overall > 2:
        #     return False
        #
        # worest_response = len([q for q in questions if q != 0] )
        #
        # if worest_response / len(questions) > 0.39:  ## less than 30% 1 or 5 in each part to get a total 1.
        #     return False
        # else:
        #     return True


    overallLeft = int(anwlist[0])
    overallright = int(anwlist[1])

    anwlist = anwlist[2:]
    segLength = int(len(anwlist)/2)
    leftList = list(map(lambda x:int(x),anwlist[:segLength]))
    rightList = list(map(lambda x:int(x),anwlist[segLength:]))


    leftFeelsGood = _singlePosition(overallLeft,leftList)
    rightFeelsGood = _singlePosition(overallright,rightList)
    #
    return (leftFeelsGood and rightFeelsGood)



if __name__ == "__main__":
    getSizeV8(Response_v8_1)
    getSizeV8(Response_v8_2)
