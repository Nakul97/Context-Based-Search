import numpy as np
import operator

productList = {"FitBit": [0,0.5,0,0.5,0,0]}

userProfile = np.load("userProfile.npy")

searchIndex = {}

for key in productList.keys():
    similarity = productList[key][0] * userProfile[0] + productList[key][1] * userProfile[1] + productList[key][2] * userProfile[2] + productList[key][3] * userProfile[3] + productList[key][4] * userProfile[4] + productList[key][5] * userProfile[5] 
    searchIndex[key] = similarity

sorted_search = sorted(searchIndex.items(), key=operator.itemgetter(1))
print(sorted_search)
