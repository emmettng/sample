# ##---------------------comparing and analsying -------------------###
#     stageMarker_v81 = stageMarker_v1
#     stageMarker_v81["resVersion"] = "v81"
#     stage1_v81 = stageOne(gender,footFeatures,shoeFeatures,stageMarker_v81)
#     v81_ids, v81_datset = _normalTrans(stage1_v81,stageMarker_v81)
#     print ("")
#
# check_dict_v81 = _modelscheckDict(v81_datset,stageMarker_v81)
#
#     # for ind,v in check_dict_v81.items():
#     #     print (ind)
#
#
#     reportArray = list()
#     verifyArray = list()
#     reportRagne = range(len(check_dict_v81.keys()))
#     for r in reportRagne:
#         for k,v in check_dict_v81.items():
#             rindex = int(k)
#             if rindex != r:
#                 continue
#             print (k)
#             reportArray.append(v)
#             ttlist = list(map(_ttF,list(zip(v81_datset[1],v))))
#             verifyArray.append(ttlist)
#
#
#     verificationSum = np.sum(verifyArray, axis=0)
#     print(len(verificationSum))
#     plt.hist(verificationSum, bins=range(30))
#     print(len(verificationSum))
#
#     prefectList = [v81_ids[i] for i, p in enumerate(verificationSum) if p == len(check_dict_v81.keys())]
#     loserList = [v81_ids[i] for i, p in enumerate(verificationSum) if p == 0]
#     print(len(prefectList))
#     print(len(loserList))
#     # plt.show()
#     print("perfect phone")
#     perfectPhone = (list(set([p[0] for p in prefectList])))
#     print(perfectPhone)
#     print("perfect sku")
#     perfectSKU = (list(set([p[1] for p in prefectList])))
#     print(perfectSKU)
#     print("loser phone")
#     loserPhone = (list(set([p[0] for p in loserList])))
#     print(loserPhone)
#     print("loser sku")
#     loserSKU = (list(set([p[1] for p in loserList])))
#     print(loserSKU)
#     totalSKU = (list(set([ids[1] for ids in v81_ids])))
#     print ("total SKU")
#     print (totalSKU)
#     totalPhone = (list(set([ids[0] for ids in v81_ids])))
#
#     joinPhone = [p for p in perfectPhone if p in loserPhone]
#     print("")
#     print ("joinPhone")
#     print(joinPhone)
#
#     joinSKU = [p for p in perfectSKU if p in loserSKU]
#     print ("join SKU")
#     print(joinSKU)
#
#     print ("sku total : %d" %len(totalSKU))
#     print ("sku perfect: %d" %len(perfectSKU))
#     print ("sku loser:   %d" %len(loserSKU))
#     print ("sku join :   %d" %len(joinSKU))
#     print ("")
#     print ("phone total : %d"  %len(totalPhone))
#     print ("phone perfect: %d" %len(perfectPhone))
#     print ("phone loser:   %d" %len(loserPhone))
#     print ("phone join :   %d" %len(joinPhone))
#
#     purePerfectPhone = [p for p in perfectPhone if p not in loserPhone]
#     pureLoserPhon = [p for p in loserPhone if p not in perfectPhone]
#     print ("pure perfect phone %d " %len(purePerfectPhone))
#     print ("pure loser phone %d " % len(pureLoserPhon))
#
#     sDict = dict()
#     for i,v in enumerate(verificationSum):
#         sku = v81_ids[i][1]
#         if v != 0 :continue
#         tmp = sDict.get(sku,0)
#         tmp+=1
#         sDict[sku] = tmp
#     print (sDict)
#
#     import operator
#     sorted_x = sorted(sDict.items(), key=operator.itemgetter(1))
#     print (sorted_x)
#
#     pDict = dict()
#     for i,v in enumerate(verificationSum):
#         phone = v81_ids[i][0]
#         if v != 0 :continue
#         if phone in perfectPhone: continue
#         tmp = pDict.get(phone,0)
#         tmp+=1
#         pDict[phone] = tmp
#     sorted_p = sorted(pDict.items(), key=operator.itemgetter(1))
#     print (sorted_p)
#     print (len(sorted_p))
#
#     for i in v81_ids:
#         phone = i[0]
#         if phone in pureLoserPhon:
#             print (i)
##=========================== correlation =================++###
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
# workbook.close()
