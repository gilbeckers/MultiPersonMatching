import readOpenPoseJson
import pose_matcher
import glob

def calc_match(modelFoto, inputFoto):
    # 2D array. Elke entry is een coordinatenkoppel(joint-point) in 3D , z-coordinaat wordt nul gekozen want we werken in 2D
    # Bevat 18 coordinatenkoppels want openpose returnt 18 joint-points
    primary = readOpenPoseJson.readJsonFile('json_data/' + modelFoto + '.json')
    secondary = readOpenPoseJson.readJsonFile('json_data/' + inputFoto + '.json')

    #parameters van functie zijn slecht gekozen => secondary = input  &&   primary = model
    result = calcEucldError.norm_cte(secondary, primary)

    return result


def calc_match_fullpath(model, input):
    model = readOpenPoseJson.readJsonFile(model)
    input = readOpenPoseJson.readJsonFile(input)

    #parameters van functie zijn slecht gekozen => secondary = input  &&   primary = model
    result = pose_matcher.norm_cte_decide_match_or_not(model, input)

    #print("Match or not: ", result)
    return result

pose = "pose1"
'''
for json in glob.iglob("/home/jochen/Documents/thesis/poses/pose1/json/*.json"):
'''

print(calc_match_fullpath("/home/jochen/Documents/thesis/poses/pose1/json/1.json", "/home/jochen/Documents/thesis/poses/pose1/json/2.json"))
print("script finished ")
