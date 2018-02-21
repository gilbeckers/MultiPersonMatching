import parse_openpose_json
import prepocessing
import affine_transformation
import pose_match
'''
Some plotties for a blog
'''

json_data_path = 'data/json_data/'
images_data_path = 'data/image_data/'

model = "3"
input = "midget1"
model_json = json_data_path + model + '.json'
input_json = json_data_path + input + '.json'

model_image = images_data_path + model + '.png'
input_image = images_data_path + input + '.jpg'

model_features = parse_openpose_json.parse_JSON_single_person(model_json)
input_features = parse_openpose_json.parse_JSON_single_person(input_json)

input_features = prepocessing.unpad(input_features)
model_features = prepocessing.unpad(model_features)

(input_trans, A ) = affine_transformation.find_transformation(model_features, input_features)
pose_match.plot_match(model_features, input_features, input_trans, model_image, input_image)

