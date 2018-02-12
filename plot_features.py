import pose_match
import matplotlib.pyplot as plt
import parse_openpose_json
import logging
import algorithmia_client

logger = logging.getLogger("pose_match")
json_data_path = 'data/json_data/'
images_data_path = 'data/image_data/'


model = "model2"
input = "model2" #""jochen_foto1_old"
model_json = json_data_path + model + '.json'
input_json = json_data_path + input + '.json'
model_image = images_data_path + model + '.png'
input_image = images_data_path + input + '.png'
model_features = parse_openpose_json.parse_JSON_single_person(model_json)
input_features = parse_openpose_json.parse_JSON_single_person(input_json)

#algorithmia_client.upload_pose_img("jochen_foto1.jpg")
#result_api = algorithmia_client.call_pose_estimation_img_on_server("p2.jpg")
#input_features = parse_openpose_json.parse_JSON_single_person_as_json(result_api)

markersize = 3
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
ax1.imshow(plt.imread(input_image))
ax1.set_title(input_image + '(input)')
ax1.plot(*zip(*input_features), marker='o', color='r', ls='', label='model', ms=markersize)  # ms = markersize

ax2.set_title(model_image + '(model)')
ax2.imshow(plt.imread(model_image))
ax2.plot(*zip(*model_features), marker='o', color='r', ls='', label='model', ms=markersize)  # ms = markersize

# ax3.set_title('(model sumperimposed on input)')
# ax3.imshow(plt.imread(input_image))
# ax3.plot(*zip(*Z), marker='o', color='r', ls='', label='model', ms=markersize)  # ms = markersize
# plt.show()
plt.show()

