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
model = "sitting6"
input = "sitting7"
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

#-------------------Simple case; poses are not checked on relation in space-------------------
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
#logger.info("$$$$$ Multi pose without norm (plotting) $$$$$$")
#pose_match.multi_person2(model_features, input_features, model_image, input_image,False) # without normalisation


