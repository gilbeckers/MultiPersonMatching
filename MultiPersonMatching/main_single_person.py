import multiperson_match
import singleperson_match
import parse_openpose_json
import numpy as np
import logging
import prepocessing
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("pose_match")
json_data_path = 'data/json_data/'
images_data_path = 'data/image_data/'
path = '/media/jochen/2FCA69D53AB1BFF49/dataset/poses/'
pose= "pose5"
data = '/media/jochen/2FCA69D53AB1BFF49/dataset/data/'
matched = '/media/jochen/2FCA69D53AB1BFF49/dataset/matched/'
'''
-------------------------------- SINGLE PERSON -------------------------------------------
Read openpose output and parse body-joint points into an 2D array of 18 rows
Elke entry is een coordinatenkoppel(joint-point) in 3D , z-coordinaat wordt nul gekozen want we werken in 2D
'''

model = "370"
input = "293"
model_json = path +pose+"/json/" +model + '.json'
input_json = path +pose+"/json/" +input + '.json'

model_image = path +pose+"/fotos/"+ model + '.png'
input_image = path +pose+"/fotos/"+ input + '.png'

model_features = parse_openpose_json.parse_JSON_single_person(model_json)
input_features = parse_openpose_json.parse_JSON_single_person(input_json)


'''
Calculate match fo real (incl. normalizing)
'''
#TODO: edit return tuple !!
match_result = singleperson.single_person(model_features, input_features, True)
logger.info("--Match or not: %s ", str(match_result.match_bool))


'''
Calculate match + plot the whole thing
'''
# Reload features bc model_features is a immutable type  -> niet meer nodig want er wordt een copy gemaalt in single_psoe()
# and is changed in single_pose in case of undetected bodyparts
#model_features = parse_openpose_json.parse_JSON_single_person(model_json)
#input_features = parse_openpose_json.parse_JSON_single_person(input_json)
singleperson_match.plot_single_person(model_features, input_features, model_image, input_image)
