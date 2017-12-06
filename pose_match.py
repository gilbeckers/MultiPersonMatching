import normalising
import prepocessing
import affine_transformation
import pose_comparison
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import logging
import numpy as np

logger = logging.getLogger(__name__)

class MatchCombo(object):

    def __init__(self, error_score, input_id, model_id, input_transformation):
        self.error_score = error_score
        self.input_id = input_id
        self.model_id = model_id
        self.input_transformation = input_transformation

#Takes two parameters, model name and input name.
#Both have a .json file in json_data and a .jpg or .png in image_data
def single_person(model_features, input_features):

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

    #TODO: construct a solid score algorithm
    error_score = (max_euclidean_error_torso + max_euclidean_error_legs)/2.0

    #TODO: wrap transformation of input back in one whole matrix
    input_transformation = np.zeros(2)
    return ( (result_torso and result_legs), error_score, input_transformation)


#Plot the calculated transformation on the model image
#And some other usefull plots for debugging
#NO NORMALIZING IS DONE HERE BECAUSE POINTS ARE PLOTTED ON THE ORIGINAL PICTURES!
def plot_single_person(model_features, input_features, model_image_name, input_image_name):
    # plot vars
    markersize = 3

    #Load images
    model_image = plt.imread(model_image_name)
    input_image = plt.imread(input_image_name)

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


'''
Description multi_person():
This function is used in the first (simple) case: MODELS SEPARATELY (mutual orientation of different poses is not checked in this simple case)
    The human poses in the image have no relation with each other and they are considered separately 
    Foreach input pose (in input_features) a matching model pose (in models_features) is searched
    Only if for each input model matches with on of the models in models_features, a global match is achieved. 

Parameters:
@:param models_poses: Takes an array of models as input because every pose that needs to be mimic has it's own model
@:param input_poses: The input is one json file. This represents an image of multiple persons and they each try to mimic one of the poses in model
'''
def multi_person(models_poses, input_poses):
    logger.info(" Multi-person matching...")
    logger.info(" amount of models: %d", len(models_poses))
    logger.info(" amount of inputs: %d", len(input_poses))
    # Some safety checks first ..
    if(len(input_poses)== 0 or len(models_poses) == 0):
        logger.info(" Multi person match failed. Inputposes or modelposes are empty")
        return False

    # Then check if there are equal or more input poses than model poses
    # We amount of input poses is smaller than model poses a global match FAILED is decided
    # Note: amount of models may be smaller than amount of input models:
    #       -> ex in case of people on the background which are also detected by openpose => background noise
    #       -> TODO: ? foreground detection necessary (in case less models than there are inputs)  NOT SURE ...
    if(len(input_poses) < len(models_poses)):
        logger.info(" Multi person match failed. Amount of input poses < model poses")
        return False

    # List of MatchCombo objects: each input has a 1 or 0 matches with a model (this match is wrapped in a MatchCombo object)
    # If at the end of the whole matching iteration, there is a MatchCombo == None => a input is failed to match whith the possible modelposes SO global match failed
    list_of_all_matches = []

    # The MatchCombo which links a inputpose with a matching modelpose
    # This is what we want to maximise ie: the best match of all possible matches found
    best_match_combo = None

    #Iterate over the input poses
    counter_input_pose = 1
    for input_pose in input_poses:
        counter_model_pose = 1

        for model_pose in models_poses:
            # Do single pose matching
            (result_match, error_score, input_transformation) = single_person(model_pose, input_pose)

            if (result_match):
                match_combo = MatchCombo(error_score, counter_input_pose, counter_model_pose, input_transformation)
                logger.info(" Match: %s InputPose(%d)->ModelPose(%d)", result_match, counter_input_pose, counter_model_pose)

                # If current MatchCombo object is empty, init it
                if best_match_combo is None:
                    best_match_combo = match_combo
                # If new match is better than current best_match, overwrite
                elif best_match_combo.error_score > match_combo.error_score:
                    best_match_combo = match_combo

            counter_model_pose = counter_model_pose + 1

        # If still no match is found (after looping over all the models); break and report a global matching failure
        if(best_match_combo  is None):
            logger.info(" MATCH FAILED. No match found for inputpose(%d). Quiting multi_person_matching, further inputposes are not considered ", counter_input_pose)
            return False

        # After comparing every possible models with a inputpose, append to match_list
        list_of_all_matches.append(best_match_combo)
        # And reset best_match field for next iteration
        best_match_combo = None

        counter_input_pose = counter_input_pose + 1


    for i in list_of_all_matches:
        if i is not None:
            print("jeej: " , i.input_id , "best match: " , i.model_id)

    return True

