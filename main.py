import pose_match
import parse_openpose_json


json_data_path = 'data/json_data/'
images_data_path = 'data/image_data/'
model = "foto1"
input = "jochen_foto1"

model_json = json_data_path + model + '.json'
input_json = json_data_path + input + '.json'

model_image = images_data_path + model + '.jpg'
input_image = images_data_path + input + '.jpg'

parse_openpose_json.parse_JSON_multi_person(json_data_path + "jochen_rob.json")

# Read openpose output and parse body-joint points into an 2D array of 18 rows
# Elke entry is een coordinatenkoppel(joint-point) in 3D , z-coordinaat wordt nul gekozen want we werken in 2D
model_features = parse_openpose_json.parse_JSON_single_person(model_json)
input_features = parse_openpose_json.parse_JSON_single_person(input_json)


#Calculate match fo real (incl. normalizing)
#print("\n--Match or not: " + str(pose_match.single_person(model_features, input_features)))

#Calculate match + plot the whole thing
pose_match.plot_single_person(model_features, input_features, model_image, input_image)

