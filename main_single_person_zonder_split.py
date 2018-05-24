import pose_match
import parse_openpose_json
import numpy as np
import logging
import prepocessing
import matplotlib.pyplot as plt
import draw_humans

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("pose_match")
json_data_path = 'data/json_data/'
images_data_path = 'data/image_data/'

json_data_path = 'data/json_data/'
images_data_path = 'data/image_data/'


model = "trap7"
input = "trap9"

model = "foto1"
input = "kleuter8"
model_json = json_data_path + model + '.json'
input_json = json_data_path + input + '_keypoints.json'

model_image = images_data_path + model + '.jpg'
input_image = images_data_path + input + '.jpg'



model_features = parse_openpose_json.parse_JSON_single_person(model_json)
input_features = parse_openpose_json.parse_JSON_single_person(input_json)


'''
Calculate match fo real (incl. normalizing)
'''
#TODO: edit return tuple !!
match_result = pose_match.single_person_zonder_split(model_features, input_features, True)
logger.info("--Match or not: %s    score= %f", str(match_result.match_bool), match_result.error_score)


'''
Calculate match + plot the whole thing
'''
# Reload features bc model_features is a immutable type  -> niet meer nodig want er wordt een copy gemaalt in single_psoe()
# and is changed in single_pose in case of undetected bodyparts
#model_features = parse_openpose_json.parse_JSON_single_person(model_json)
#input_features = parse_openpose_json.parse_JSON_single_person(input_json)
pose_match.plot_single_person_zonder_split(model_features, input_features, model_image, input_image)

plt.show()


