import pose_match
import parse_openpose_json
import numpy as np
import logging
import prepocessing
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("pose_match")
json_data_path = 'data/json_data/'
images_data_path = 'data/image_data/'

'''
-------------------------------- SINGLE PERSON -------------------------------------------
Read openpose output and parse body-joint points into an 2D arr ay of 18 rows
Elke entry is een coordinatenkoppel(joint-point) in 3D , z-coordinaat wordt nul gekozen want we werken in 2D
'''

model = "model4"
input = "pose4_fout"
model_json = json_data_path + model + '.json'
input_json = json_data_path + input + '.json'

model_image = images_data_path + model + '.png'
input_image = images_data_path + input + '.png'

model_features = parse_openpose_json.parse_JSON_single_person(model_json)
input_features = parse_openpose_json.parse_JSON_single_person(input_json)

'''
Calculate match fo real (incl. normalizing)
'''
#TODO: edit return tuple !!
match_result = pose_match.single_person(model_features, input_features, True)
logger.info("--Match or not: %s ", str(match_result.match_bool))


'''
Calculate match + plot the whole thing
'''
# Reload features bc model_features is a immutable type  -> niet meer nodig want er wordt een copy gemaalt in single_psoe()
# and is changed in single_pose in case of undetected bodyparts
#model_features = parse_openpose_json.parse_JSON_single_person(model_json)
#input_features = parse_openpose_json.parse_JSON_single_person(input_json)
pose_match.plot_single_person(model_features, input_features, model_image, input_image)



