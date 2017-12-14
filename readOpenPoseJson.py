import json
from pprint import pprint
import numpy

def readJsonFile(filename):
    with open(filename) as data_file:
        data = json.load(data_file)

    #pprint(data)


    #Keypoints
    keypointsPeople1 = data["people"][0]["pose_keypoints"] #enkel 1 persoon  => [0]

    #18 3D coordinatenkoppels (joint-points)
    array = numpy.zeros((18,3))

    arrayIndex = 0
    for i in range(0, len(keypointsPeople1), 3):
        array[arrayIndex][0] = keypointsPeople1[i]
        array[arrayIndex][1] = keypointsPeople1[i+1]
        arrayIndex+=1

    return array