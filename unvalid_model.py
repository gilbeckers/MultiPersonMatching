import pose_match
import parse_openpose_json
import numpy as np
import logging
import prepocessing
import matplotlib.pyplot as plt

'''
-------------------------------- UNVALID MODELPOSES TEST CASE -------------------------------------------
Testjes voor unvalid model poses (single person)

'''

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("pose_match")
json_data_path = 'data/json_data/'
images_data_path = 'data/image_data/'


model = "1027"
input = "846"
model_json = json_data_path + model + '.json'
input_json = json_data_path + input + '.json'
model_image = images_data_path + model + '.png'
input_image = images_data_path + input + '.png'
model_features = parse_openpose_json.parse_JSON_single_person(model_json)
input_features = parse_openpose_json.parse_JSON_single_person(input_json)

model_features[2] = [0,0]

# With valid input and valid model
match_result = pose_match.single_person(model_features, input_features, True)
logger.info("--Match or not: %s ", str(match_result.match_bool))


print("model: " , model_features)
print("input: " , input_features)

pose_match.plot_single_person(model_features, input_features, model_image, input_image,
                              "valid input", "unvalid model",
                              "match result: " + str(match_result.match_bool) + "(" + str(round(match_result.error_score, 5)) + ")")


plt.show()