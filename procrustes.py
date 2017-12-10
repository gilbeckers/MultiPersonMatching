from scipy.spatial import procrustes
import matplotlib.pyplot as plt
import numpy as np
import parse_openpose_json
import prepocessing
import procrustes3

input_name = "midget1"
model_name = "spreid1"

model = parse_openpose_json.parse_JSON_single_person("data/json_data/"+model_name+".json")
input = parse_openpose_json.parse_JSON_single_person("data/json_data/"+input_name+".json")

input_image = "data/image_data/"+input_name+".jpg"
model_image = "data/image_data/"+model_name+".jpg"


input = prepocessing.unpad(input)
model = prepocessing.unpad(model)

(d, Z, m)=procrustes3.procrustes(model, input, False)


# Zoeken naar laagste punt van lichaam (linker of rechter voet)
# =>   Max van linker en recht voet (y-coordinaat)
# => pose wordt aligned met laagste punt (puur translatie)
voet_index = None

if model[10][1] >= model[13][1]:
    voet_index = 10
else:
    voet_index = 13

print("laagste punt: " , model[voet_index][1])

translatie_factor = model[voet_index][1] - Z[voet_index][1]
print("trans: ", translatie_factor)

Z[:,1] = Z[:,1] + translatie_factor



print(Z)
print(m)


markersize = 3
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
ax1.imshow(plt.imread(input_image))
ax1.plot(*zip(*input), marker='o', color='r', ls='', label='model', ms=markersize)  # ms = markersize
ax2.imshow(plt.imread(model_image))
ax2.plot(*zip(*model), marker='o', color='r', ls='', label='model', ms=markersize)  # ms = markersize
ax3.imshow(plt.imread(model_image))
ax3.plot(*zip(*Z), marker='o', color='r', ls='', label='model', ms=markersize)  # ms = markersize
plt.show()
