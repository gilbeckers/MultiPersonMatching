import json
import numpy

#parse the openpose json output and output an array of 18 rows (body -joint points / keypoints)
# so undetected body parts (openpose errors, labeled by openpose as (0,0) )
#   -> stay (0,0) and can be identified in this way
def parse_JSON_single_person(filename):
    with open(filename) as data_file:
        data = json.load(data_file)

    #Keypoints
    keypointsPeople1 = data["people"][0]["pose_keypoints"] #enkel 1 persoon  => [0]

    #18 2D coordinatenkoppels (joint-points)
    array = numpy.zeros((18,2))
    #list = []
    arrayIndex = 0

    for i in range(0, len(keypointsPeople1), 3):
        array[arrayIndex][0] = keypointsPeople1[i]
        array[arrayIndex][1] = keypointsPeople1[i+1]
        arrayIndex+=1

        #feature = [keypointsPeople1[i], keypointsPeople1[i+1]]
        #list.append(feature)

    return array
    #return list

def parse_JSON_single_person_as_json(filename):
    #data = json.load(filename)
    data = filename
    #Keypoints
    keypointsPeople1 = data["people"][0]["pose_keypoints"] #enkel 1 persoon  => [0]

    #18 2D coordinatenkoppels (joint-points)
    array = numpy.zeros((18,2))
    #list = []
    arrayIndex = 0

    for i in range(0, len(keypointsPeople1), 3):
        array[arrayIndex][0] = keypointsPeople1[i]
        array[arrayIndex][1] = keypointsPeople1[i+1]
        arrayIndex+=1

        #feature = [keypointsPeople1[i], keypointsPeople1[i+1]]
        #list.append(feature)

    return array
    #return list

def parse_JSON_multi_person(filename):
    with open(filename) as data_file:
        data = json.load(data_file)

    list_of_features = []

    keypoints = data["people"]
    for k in range(0, len(keypoints)):
        person_keypoints = keypoints[k]["pose_keypoints"]

        # 18 3D coordinatenkoppels (joint-points)
        array = numpy.zeros((18, 2))
        arrayIndex = 0
        for i in range(0, len(person_keypoints), 3):
            array[arrayIndex][0] = person_keypoints[i]
            array[arrayIndex][1] = person_keypoints[i + 1]
            arrayIndex += 1
        list_of_features.append(array)

    return list_of_features

def parse_JSON_multi_person_as_json(filename):
    data = filename

    list_of_features = []

    keypoints = data["people"]
    for k in range(0, len(keypoints)):
        person_keypoints = keypoints[k]["pose_keypoints"]

        # 18 3D coordinatenkoppels (joint-points)
        array = numpy.zeros((18, 2))
        arrayIndex = 0
        for i in range(0, len(person_keypoints), 3):
            array[arrayIndex][0] = person_keypoints[i]
            array[arrayIndex][1] = person_keypoints[i + 1]
            arrayIndex += 1
        list_of_features.append(array)

    return list_of_features

