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

    def __init__(self, error_score, input_id, model_id, model_features, input_features, input_transformation):
        self.error_score = error_score
        self.input_id = input_id
        self.model_id = model_id
        self.model_features = model_features # niet noodzaakelijk voor logica, wordt gebruikt voor plotjes
        self.input_features = input_features # same
        self.input_transformation = input_transformation

#Takes two parameters, model name and input name.
#Both have a .json file in json_data and a .jpg or .png in image_data
def single_person(model_features, input_features, normalise):

    #Normalise features: crop => delen door Xmax & Ymax (NIEUWE MANIER!!)
    if(normalise):
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
    eucl_dis_tresh_torso = 0.06
    rotation_tresh_torso = 55
    eucl_dis_tresh_legs = 0.14
    rotation_tresh_legs = 40

    eucld_dis_shoulders_tresh = 0.06
    ################################

    result_torso = pose_comparison.decide_torso_shoulders_incl(max_euclidean_error_torso, transformation_matrix_torso,
                                                eucl_dis_tresh_torso, rotation_tresh_torso,
                                                max_euclidean_error_shoulders, eucld_dis_shoulders_tresh)
    result_legs = pose_comparison.decide_legs(max_euclidean_error_legs, transformation_matrix_legs,
                                              eucl_dis_tresh_legs, rotation_tresh_legs)

    #TODO: construct a solid score algorithm
    error_score = (max_euclidean_error_torso + max_euclidean_error_legs)/2.0
    input_transformation = prepocessing.unsplit(input_transformed_face, input_transformed_torso, input_transformed_legs)
    #logger.info("face: %s" , input_transformed_face)
    #logger.info("legs: %s" , input_transformed_legs)
    #logger.info("torso: %s" , input_transformed_torso)
    #logger.info("tot %s", input_transformation)
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
    ax1.plot(*zip(*model_features), marker='o', color='r', ls='', label='model', ms=markersize)  # ms = markersize
    red_patch = mpatches.Patch(color='red', label='model')
    ax1.legend(handles=[red_patch])

    ax2.set_title(input_image_name)
    ax2.imshow(input_image)
    ax2.plot(*zip(*input_features), marker='o', color='b', ls='', ms=markersize)
    ax2.legend(handles=[mpatches.Patch(color='blue', label='input')])

    whole_input_transform = prepocessing.unsplit(input_transformed_face, input_transformed_torso, input_transformed_legs)
    ax3.set_title('Transformation of input + model')
    ax3.imshow(model_image)
    ax3.plot(*zip(*model_features), marker='o', color='r', ls='', label='model', ms=markersize)  # ms = markersize
    ax3.plot(*zip(*whole_input_transform), marker='o', color='b', ls='', ms=markersize)
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
# THE NEW one: for every modelpose , a matching input is seeked
# Enkel zo kan je een GLOBAL MATCH FAILED besluiten na dat er geen matching inputpose is gevonden voor een modelpose
def multi_person(models_poses, input_poses, model_image_name, input_image_name):
    logger.info(" Multi-person matching...")
    logger.info(" amount of models: %d", len(models_poses))
    logger.info(" amount of inputs: %d", len(input_poses))
    # Some safety checks first ..
    if(len(input_poses)== 0 or len(models_poses) == 0):
        logger.info(" Multi person match failed. Inputposes or modelposes are empty")
        return False

    # Then check if there are equal or more input poses than model poses
    # When the amount of input poses is smaller than model poses a global match FAILED is decided
    # Note: amount of models may be smaller than amount of input models:
    #       -> ex in case of people on the background which are also detected by openpose => background noise
    #       -> TODO: ? foreground detection necessary (in case less models than there are inputs)  NOT SURE ...
    if(len(input_poses) < len(models_poses)):
        logger.info(" Multi person match failed. Amount of input poses < model poses")
        return False

    # When there are more inputposes than modelsposes, print warning
    # I this case pose match still needs to proceed,
    #  BUT it's possible that a background pose is matched with one of the proposed modelposes!! (something we don't want)
    if (len(input_poses) > len(models_poses)):
        logger.info(" !! WARNING !! Amount of input poses > model poses")
        #return False

    # List of MatchCombo objects: each model has a 0 or 1 or more matches with a input (this match is wrapped in a MatchCombo object)
    #   -> we only allow the case with 1 match!
    #   1. cases with more than 1 matches are reduces with only 1 match (the best-match)

    #   2. case with 0 match result in a GLOBAL MATCH FAILED; because the modelpose is not found in the input
    #      If at the end of the whole matching iteration,
    #      there is a MatchCombo == None => a model is failed to match whith the possible inputposes SO global match failed
    #
    list_of_all_matches = []

    # The MatchCombo which links a modelpose with a matching inputpose
    # This is what we want to maximise ie: the best match of all possible matches found
    best_match_combo = None

    #Iterate over the model poses
    counter_model_pose = 1
    for model_pose in models_poses:
        counter_input_pose = 1

        for input_pose in input_poses:
            # Do single pose matching
            (result_match, error_score, input_transformation) = single_person(model_pose, input_pose, True)

            if (result_match):
                match_combo = MatchCombo(error_score, counter_input_pose, counter_model_pose, model_pose, input_pose, input_transformation)
                logger.info(" Match: %s ModelPose(%d)->InputPose(%d)", result_match, counter_model_pose, counter_input_pose)

                # If current MatchCombo object is empty, init it
                if best_match_combo is None:
                    best_match_combo = match_combo
                # If new match is better (=has a lower error_score) than current best_match, overwrite
                elif best_match_combo.error_score > match_combo.error_score:
                    best_match_combo = match_combo

            counter_input_pose = counter_input_pose + 1

        # If still no match is found (after looping over all the inputs); this model is not found in proposed inputposes
        # This can mean two things:
        #   1. The user failed to mimic on of the proposed model poses
        #   2. (FOUT)  MAG WEG DIT? This inputpose describes someone in the background,
        #       in which case this pose is considered as background noise
        #       and should not influence the global matching result

        if(best_match_combo  is None):
            logger.info(" MATCH FAILED. No match found for modelpose(%d). User failed to match a modelpose ", counter_model_pose)
            #return False

        # After comparing every possible models with a inputpose, append to match_list
        list_of_all_matches.append(best_match_combo)
        # And reset best_match field for next iteration
        best_match_combo = None

        counter_model_pose = counter_model_pose + 1


    for i in list_of_all_matches:
        if i is not None:
            #print("jeej: " , i.input_id , "best match: " , i.model_id)
            (result, error_score, input_transformation) = single_person(i.model_features, i.input_features, False)
            #print("transss: " , input_transformation)
            plot_match(i.model_features, i.input_features, input_transformation, model_image_name, input_image_name)

    return list_of_all_matches


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
# THe old one: for every input pose , a matching model is seeked
def multi_person_old(models_poses, input_poses, model_image_name, input_image_name):
    logger.info(" Multi-person matching...")
    logger.info(" amount of models: %d", len(models_poses))
    logger.info(" amount of inputs: %d", len(input_poses))
    # Some safety checks first ..
    if(len(input_poses)== 0 or len(models_poses) == 0):
        logger.info(" Multi person match failed. Inputposes or modelposes are empty")
        return False

    # Then check if there are equal or more input poses than model poses
    # When the amount of input poses is smaller than model poses a global match FAILED is decided
    # Note: amount of models may be smaller than amount of input models:
    #       -> ex in case of people on the background which are also detected by openpose => background noise
    #       -> TODO: ? foreground detection necessary (in case less models than there are inputs)  NOT SURE ...
    if(len(input_poses) < len(models_poses)):
        logger.info(" Multi person match failed. Amount of input poses < model poses")
        return False

    # When there are more inputposes than modelsposes, print warning
    # I this case pose match still needs to proceed,
    #  BUT it's possible that a background pose is matched with one of the proposed modelposes!!
    if (len(input_poses) > len(models_poses)):
        logger.info(" !! WARNING !! Amount of input poses > model poses")
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
            (result_match, error_score, input_transformation) = single_person(model_pose, input_pose, True)

            if (result_match):
                match_combo = MatchCombo(error_score, counter_input_pose, counter_model_pose, model_pose, input_pose, input_transformation)
                logger.info(" Match: %s InputPose(%d)->ModelPose(%d)", result_match, counter_input_pose, counter_model_pose)

                # If current MatchCombo object is empty, init it
                if best_match_combo is None:
                    best_match_combo = match_combo
                # If new match is better (=has a lower error_score) than current best_match, overwrite
                elif best_match_combo.error_score > match_combo.error_score:
                    best_match_combo = match_combo

            counter_model_pose = counter_model_pose + 1

        # If still no match is found (after looping over all the models); this input is not found in proposed modelposes
        # This can mean two thins:
        #   1. The user failed to mimic on of the proposed model poses
        #   2. This inputpose describes someone in the background,
        #       in which case this pose is considered as background noise
        #       and should not influence the global matching result

        if(best_match_combo  is None):
            logger.info(" MATCH FAILED. No match found for inputpose(%d). User either failed to match a modelpose or this inputpose is background noise", counter_input_pose)
            #return False

        # After comparing every possible models with a inputpose, append to match_list
        list_of_all_matches.append(best_match_combo)
        # And reset best_match field for next iteration
        best_match_combo = None

        counter_input_pose = counter_input_pose + 1


    for i in list_of_all_matches:
        if i is not None:
            #print("jeej: " , i.input_id , "best match: " , i.model_id)
            (result, error_score, input_transformation) = single_person(i.model_features, i.input_features, False)
            #print("transss: " , input_transformation)
            plot_match(i.model_features, i.input_features, input_transformation, model_image_name, input_image_name)

    return list_of_all_matches


'''
This function is used in the second (complex) case: The models are dependent ofeach other in space
'''
def multi_person2():


    return

def plot_match(model_features, input_features, input_transform_features, model_image_name, input_image_name):
    # plot vars
    markersize = 3

    # Load images
    model_image = plt.imread(model_image_name)
    input_image = plt.imread(input_image_name)

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
    implot = ax1.imshow(model_image)
    ax1.set_title(model_image_name + '(model)')
    ax1.plot(*zip(*model_features), marker='o', color='r', ls='', label='model', ms=markersize)  # ms = markersize
    red_patch = mpatches.Patch(color='red', label='model')
    ax1.legend(handles=[red_patch])

    ax2.set_title(input_image_name)
    ax2.imshow(input_image)
    ax2.plot(*zip(*input_features), marker='o', color='b', ls='', ms=markersize)
    ax2.legend(handles=[mpatches.Patch(color='blue', label='input')])

    ax3.set_title('Transformation of input + model')
    ax3.imshow(model_image)
    #ax3.plot(*zip(*model_features), marker='o', color='y', ls='', label='model', ms=markersize)  # ms = markersize
    ax3.plot(*zip(*input_transform_features), marker='o', color='magenta', ls='', ms=markersize)
    ax3.legend(handles=[mpatches.Patch(color='yellow', label='Transformation of model')])

    plt.show()

    return
