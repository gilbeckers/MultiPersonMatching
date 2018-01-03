from scipy.constants.codata import precision

import pose_match
import parse_openpose_json
import numpy as np
import logging
import prepocessing
import dataset_actions
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
path = '/media/jochen/2FCA69D53AB1BFF49/dataset/poses/'

dataset_actions.find_treshholds("pose1")
'''
dataset_actions.replace_json_files("pose5")
dataset_actions.check_pose_data("pose5")

dataset_actions.replace_json_files("pose1")
dataset_actions.find_treshholds("pose1")
'''





'''
-------------------------------- SINGLE PERSON -------------------------------------------
Read openpose output and parse body-joint points into an 2D array of 18 rows
Elke entry is een coordinatenkoppel(joint-point) in 3D , z-coordinaat wordt nul gekozen want we werken in 2D
'''

'''

model = "foto1"
input = "midget1"
model_json = json_data_path + model + '.json'
input_json = json_data_path + input + '.json'

model_image = images_data_path + model + '.jpg'
input_image = images_data_path + input + '.jpg'

model_features = parse_openpose_json.parse_JSON_single_person(model_json)
input_features = parse_openpose_json.parse_JSON_single_person(input_json)

input_features = prepocessing.unpad(input_features)
model_features = prepocessing.unpad(model_features)



(result, error_score, input_transform) = pose_match.single_person(model_features, input_features, True)
print "--Match or not: %s " + str(result)
'''
'''
Calculate match + plot the whole thing
'''
#pose_match.plot_single_person(model_features, input_features, model_image, input_image)
