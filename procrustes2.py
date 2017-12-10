from math import cos, sin, atan
from numpy import array, dot
import matplotlib.pyplot as plt
import parse_openpose_json
import prepocessing
import matplotlib.pyplot as plt

def translate(points):
    """ This function translates the points to center around the origin. """

    return points - sum(points) / points.shape[0]

def scale(points):
    """ This function scales the points. Assumes that points are already
        centered around the origin. """

    scale = sum(pow(sum(pow(points, 2.0) / float(points.shape[0])), .5))
    return points / scale

def get_rotation(template, points):

    numerator = sum(points[:, 0] * template[:, 1] - \
                points[:, 1] * template[:, 0])

    divisor = sum(points[:, 0] * template[:, 0] + \
                points[:, 1] * template[:, 1])

    #   Avoiding dividing by zero
    if divisor == 0.0:
        divisor = 0.00000000001

    return atan(numerator / divisor)

def rotate(points, theta, center_point=(0, 0)):
    """
    Rotates the points around the center point.
    """

    new_array = array(points)

    new_array[0, :] -= center_point[0]
    new_array[1, :] -= center_point[1]

    new_array = dot(rotation_matrix_2D(theta),
                    new_array.transpose()).transpose()

    new_array[0, :] += center_point[0]
    new_array[1, :] += center_point[1]

    return new_array

def rotation_matrix_2D(theta):

    return array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

def procrustes(template, points):
    """
    This function computes the minizimed distance between the template
    and a test set of point once the test set has been scaled, translated,
    and rotated to the best fit.

    """

    tmp_points = scale(translate(points))
    tmp_template = scale(translate(template))

    theta = get_rotation(tmp_template, tmp_points)
    r_points = rotate(tmp_points, theta)

    return (pow(pow(tmp_template - r_points, 2.0).sum(), .5), r_points)




#   Define the template, just a triangle
template = array([[-60., 0.],
                  [60., 20.],
                  [60., -20.],
                  [-60., 0.]])

#   Points that are tested
points = array([[50., 170.],
                [70., 50.],
                [30., 50.],
                [50., 170.]])

input_name = "foto3"
model_name = "jochen_foto2"

input = parse_openpose_json.parse_JSON_single_person("data/json_data/"+input_name + ".json")
model = parse_openpose_json.parse_JSON_single_person("data/json_data/"+model_name + ".json")


input_image = "data/image_data/"+input_name+".jpg"
model_image = "data/image_data/"+model_name+".jpg"

input = prepocessing.unpad(input)
model = prepocessing.unpad(model)

#norm_input = scale(translate(input))
#norm_model = scale(translate(model))

#   This is will be the function that we will call
(distance, r_points) = procrustes(model, input)
print('procrustes distance:', distance)
print('input: ', input)
print("trans: " , r_points)

markersize = 3

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
#f.gca().invert_xaxis()
f.gca().invert_yaxis()

ax1.imshow(plt.imread(input_image))
ax1.plot(*zip(*input), marker='o', color='r', ls='', label='model', ms=markersize)  # ms = markersize

ax2.imshow(plt.imread(model_image))
ax2.plot(*zip(*model), marker='o', color='r', ls='', label='model', ms=markersize)  # ms = markersize

plt.figure()
plt.gca().invert_yaxis()
plt.plot(*zip(*r_points), marker='o', color='r', ls='', label='model', ms=markersize)  # ms = markersize
plt.show()
