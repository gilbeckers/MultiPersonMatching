import pose_match
import parse_openpose_json
import numpy as np
import logging
import prepocessing
import matplotlib.pyplot as plt

'''
-------------------------------- UNVALID POSES TEST CASE -------------------------------------------
Testjes voor unvalid input poses (single person)

'''

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("pose_match")
json_data_path = 'unvalid/json_data/'
images_data_path = 'unvalid/image_data/'


model = "846"
input = "1027"
model_json = json_data_path + model + '.json'
input_json = json_data_path + input + '.json'
model_image = images_data_path + model + '.png'
input_image = images_data_path + input + '.png'
model_features = parse_openpose_json.parse_JSON_single_person(model_json)
input_features = parse_openpose_json.parse_JSON_single_person(input_json)


# Some testcases ...
'''
unvalid_list = [
    [[3], "LShoulder" ],
    [[6], "RShoulder"],
    [[3,6], "L&R Shoulder"],
    [[2, 3, 6], "LElbow + L&R Shoulder"],
    [[2, 3, 6, 7], "LElbow + L&R Shoulder + RWrist"],
    [[2, 3, 6, 7, 4], "LElbow + L&R Shoulder + L&RWrist"],
]
'''

unvalid_list = [
    [[9], "RKnee" ],
    [[8], "RHip" ],
    [[9, 12], "L&R Knee" ],
    [[9, 12, 13 ], "L&R Knee + LAnkle"],
]

match_result = pose_match.single_person(model_features, input_features, True)
logger.info("--Match or not: %s ", str(match_result.match_bool))
pose_match.plot_single_person(model_features, input_features, model_image, input_image,
                              "valid input", "valid model",
                              "match result: " + str(match_result.match_bool) + "(" + str(round(match_result.error_score, 5)) + ")")

'''
Make the inputpose unvalid by altering some features to (0,0)
'''
for unvalid_test_item in unvalid_list:
    input_features_unvalid = np.copy(input_features)

    for unvalid_feature in unvalid_test_item[0]:
        # input_features_unvalid[3] = np.array([0,0])
        input_features_unvalid[unvalid_feature] = np.array([0, 0])

    match_result = pose_match.single_person(model_features, input_features_unvalid, True)
    logger.info("--Match or not: %s ", str(match_result.match_bool))
    pose_match.plot_single_person(model_features, input_features_unvalid, model_image, input_image,
                                  unvalid_test_item[1], "valid model",
                                  "match result: " + str(match_result.match_bool) + "(" + str(round(match_result.error_score, 5)) + ")")


plt.show()