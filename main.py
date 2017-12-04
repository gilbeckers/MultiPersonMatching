import pose_match

json_data_path = 'data/json_data/'
images_data_path = 'data/image_data/'
model = "foto1"
input = "foto2"

model_json = json_data_path + model + '.json'
input_json = json_data_path + input + '.json'

model_image = images_data_path + model + '.jpg'
input_image = images_data_path + input + '.jpg'

#Calculate match + plot the whole thing
pose_match.plot_single_person(model_json, input_json, model_image, input_image)

#Calculate match fo real (incl. normalizing)
print("\n--Match or not: " + str(pose_match.single_person(model_json, input_json)))

