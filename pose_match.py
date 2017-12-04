import parse_openpose_json
import normalising
import prepocessing
import affine_transformation
import pose_comparison

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



#Takes two parameters, model name and input name.
#Both have a .json file in json_data and a .jpg or .png in image_data
def single_person(model_file_name, input_file_name):
    # Read openpose output and parse body-joint points into an 2D array of 18 rows
    # Elke entry is een coordinatenkoppel(joint-point) in 3D , z-coordinaat wordt nul gekozen want we werken in 2D
    model_features = parse_openpose_json.parse_JSON_single_person(model_file_name)
    input_features = parse_openpose_json.parse_JSON_single_person(input_file_name)

    #Normalise features: crop => delen door Xmax & Ymax (NIEUWE MANIER!!)
    model_features = normalising.cut(model_features)
    input_features = normalising.cut(input_features)

    #Split features in three parts
    (model_face, model_torso, model_legs) = prepocessing.split_in_face_legs_torso(model_features)
    (input_face, input_torso, input_legs) = prepocessing.split_in_face_legs_torso(input_features)

    # Zoek transformatie om input af te beelden op model
    # Returnt transformatie matrix + afbeelding/image van input op model
    (input_transformed_face, transformation_matrix_face) = affine_transformation.find_transformation(model_face, input_face)
    (input_transformed_torso, transformation_matrix_torso) = affine_transformation.find_transformation(model_torso, input_torso)
    (input_transformed_legs, transformation_matrix_legs) = affine_transformation.find_transformation(model_legs, input_legs)

    max_euclidean_error_face = pose_comparison.max_euclidean_distance(model_face, input_transformed_face)
    max_euclidean_error_torso = pose_comparison.max_euclidean_distance(model_torso, input_transformed_torso)
    max_euclidean_error_legs = pose_comparison.max_euclidean_distance(model_legs, input_transformed_legs)

    max_euclidean_error_shoulders = pose_comparison.max_euclidean_distance_shoulders(model_torso, input_transformed_torso)


    ######### THE THRESHOLDS #######
    eucl_dis_tresh_torso = 0.05
    rotation_tresh_torso = 18
    eucl_dis_tresh_legs = 0.0395
    rotation_tresh_legs = 14

    eucld_dis_shoulders_tresh = 0.035
    ################################

    result_torso = pose_comparison.decide_torso_shoulders_incl(max_euclidean_error_torso, transformation_matrix_torso,
                                                eucl_dis_tresh_torso, rotation_tresh_torso,
                                                max_euclidean_error_shoulders, eucld_dis_shoulders_tresh)
    result_legs = pose_comparison.decide_legs(max_euclidean_error_legs, transformation_matrix_legs,
                                              eucl_dis_tresh_legs, rotation_tresh_legs)

    return result_torso and result_legs


#Plot the calculated transformation on the model image
#And some other usefull plots for debugging
#NO NORMALIZING IS DONE HERE BECAUSE POINTS ARE PLOTTED ON THE ORIGINAL PICTURES!
def plot_single_person(model_json, input_json, model_image_name, input_image_name):
    # plot vars
    markersize = 3

    #Load images
    model_image = plt.imread(model_image_name)
    input_image = plt.imread(input_image_name)

    # Read openpose output and parse body-joint points into an 2D array of 18 rows
    # Elke entry is een coordinatenkoppel(joint-point) in 3D , z-coordinaat wordt nul gekozen want we werken in 2D
    model_features = parse_openpose_json.parse_JSON_single_person(model_json)
    input_features = parse_openpose_json.parse_JSON_single_person(input_json)

    # Split features in three parts
    (model_face, model_torso, model_legs) = prepocessing.split_in_face_legs_torso(model_features)
    (input_face, input_torso, input_legs) = prepocessing.split_in_face_legs_torso(input_features)

    # Zoek transformatie om input af te beelden op model
    # Returnt transformatie matrix + afbeelding/image van input op model
    (input_transformed_face, transformation_matrix_face) = affine_transformation.find_transformation(model_face,
                                                                                                     input_face)
    (input_transformed_torso, transformation_matrix_torso) = affine_transformation.find_transformation(model_torso,
                                                                                                       input_torso)
    (input_transformed_legs, transformation_matrix_legs) = affine_transformation.find_transformation(model_legs,
                                                                                                     input_legs)
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
    implot = ax1.imshow(model_image)
    ax1.set_title(model_image_name + '(model)')
    ax1.plot(*zip(*model_torso), marker='o', color='r', ls='', label='model', ms=markersize)  # ms = markersize
    ax1.plot(*zip(*model_legs), marker='o', color='r', ls='', label='model', ms=markersize)  # ms = markersize
    ax1.plot(*zip(*model_face), marker='o', color='r', ls='', label='model', ms=markersize)  # ms = markersize
    red_patch = mpatches.Patch(color='red', label='model')
    ax1.legend(handles=[red_patch])

    ax2.set_title(input_image_name)
    ax2.imshow(input_image)
    ax2.plot(*zip(*input_torso), marker='o', color='b', ls='', ms=markersize)
    ax2.plot(*zip(*input_legs), marker='o', color='b', ls='', ms=markersize)
    ax2.plot(*zip(*input_face), marker='o', color='b', ls='', ms=markersize)
    ax2.legend(handles=[mpatches.Patch(color='blue', label='input')])

    ax3.set_title('Transformation of input + model')
    ax3.imshow(model_image)
    ax3.plot(*zip(*model_torso), marker='o', color='r', ls='', label='model', ms=markersize)  # ms = markersize
    ax3.plot(*zip(*model_legs), marker='o', color='r', ls='', label='model', ms=markersize)  # ms = markersize
    ax3.plot(*zip(*model_face), marker='o', color='r', ls='', label='model', ms=markersize)  # ms = markersize
    ax3.plot(*zip(*input_transformed_torso), marker='o', color='y', ls='', ms=markersize)
    ax3.plot(*zip(*input_transformed_legs), marker='o', color='y', ls='', ms=markersize)
    ax3.plot(*zip(*input_transformed_face), marker='o', color='y', ls='', ms=markersize)
    ax3.legend(handles=[mpatches.Patch(color='yellow', label='Transformation of model')])

    plt.show()

