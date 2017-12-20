import pose_match
import parse_openpose_json
import numpy as np
import logging
import prepocessing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pose_match")
json_data_path = 'data/json_data/'
images_data_path = 'data/image_data/'

'''
-------------------- MULTI PERSON -------------------------------------
'''
model = "duo3"
input = "duo4"
model_json = json_data_path + model + '.json'
input_json = json_data_path + input + '.json'
model_image = images_data_path + model + '.jpg'
input_image = images_data_path + input + '.jpg'

model_features = parse_openpose_json.parse_JSON_multi_person(model_json)
input_features = parse_openpose_json.parse_JSON_multi_person(input_json)


#model1 = model_features[0]
#models_array = [np.array(model1)]
#model2 = model_features[1]
#models_array = [np.array(model1), np.array(model2)]


models_array = model_features

#Simple case; poses are not checked on relation in space
#pose_match.multi_person(models_array, input_features, model_image, input_image)

#--------------------Second case; poses ARE checked on relation in space WITH NORMALISATION------------

logger.info("$$$$$ Multi pose with normalisation $$$$$$")
pose_match.multi_person2(models_array, input_features, model_image, input_image) # with normalisation



#--------------------Second case; poses ARE checked on relation in space ZONDER NORMALISATION---------
# -> Plotjesssss
# DISCLAIMER: It normal that the plot is fucked up in case of undetected body parts in the input
#  -> this is because no normalisation is done here (because of the plots)
#     and thus these sneaky (0,0) points are not handled.
# TODO: maybe not include the (0,0) handler only the normalising part??
# TODO: !!!!!! problem when input pose includes undetected body feature!!!
model_features = parse_openpose_json.parse_JSON_multi_person(model_json)
input_features = parse_openpose_json.parse_JSON_multi_person(input_json)
logger.info("$$$$$ Multi pose without norm (plotting) $$$$$$")
#pose_match.multi_person2(model_features, input_features, model_image, input_image,False) # without normalisation

'''
-------------------------------- SINGLE PERSON -------------------------------------------
Read openpose output and parse body-joint points into an 2D array of 18 rows
Elke entry is een coordinatenkoppel(joint-point) in 3D , z-coordinaat wordt nul gekozen want we werken in 2D
'''

model = "duo3"
input = "duo4"
model_json = json_data_path + model + '.json'
input_json = json_data_path + input + '.json'

model_image = images_data_path + model + '.jpg'
input_image = images_data_path + input + '.jpg'

model_features = parse_openpose_json.parse_JSON_single_person(model_json)
input_features = parse_openpose_json.parse_JSON_single_person(input_json)

#input_features = prepocessing.unpad(input_features)
#model_features = prepocessing.unpad(model_features)


'''
Calculate match fo real (incl. normalizing)
'''
#logger.info("model before: %s", str(model_features))
#TODO: edit return tuple !!
#match_result = pose_match.single_person(model_features, input_features, True)
#logger.info("--Match or not: %s ", str(match_result.match_bool))


'''
Calculate match + plot the whole thing
'''
# Reload features bc model_features is a immutable type  -> niet meer nodig want er wordt een copy gemaalt in single_psoe()
# and is changed in single_pose in case of undetected bodyparts
#model_features = parse_openpose_json.parse_JSON_single_person(model_json)
#input_features = parse_openpose_json.parse_JSON_single_person(input_json)
#logger.info("model after: %s", str(model_features))

#pose_match.plot_single_person(model_features, input_features, model_image, input_image)



