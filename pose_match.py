import collections
import normalising
import prepocessing
import affine_transformation
import pose_comparison
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import logging
import numpy as np
import proc_do_it

logger = logging.getLogger("pose_match")

# Init the returned tuple
MatchResult = collections.namedtuple("MatchResult", ["match_bool", "error_score", "input_transformation"])

class MatchCombo(object):

    def __init__(self, error_score, input_id, model_id, model_features, input_features, input_transformation):
        self.error_score = error_score
        self.input_id = input_id
        self.model_id = model_id
        self.model_features = model_features # niet noodzaakelijk voor logica, wordt gebruikt voor plotjes
        self.input_features = input_features # same
        self.input_transformation = input_transformation

'''
Description single_person():
Takes two parameters, model name and input name.
Both have a .json file in json_data and a .jpg or .png in image_data

Parameters:
@:param model_features: 
@:param input_features: 

Returns:
@:returns result matching
@:returns error_score
@:returns input_transformation
'''
def single_person(model_features, input_features, normalise=True):
    # TODO: Make a local copy ??
    # Because np.array is a mutable type => passed by reference
    #   -> dus als model wordt verandert wordt er met gewijzigde array
    #       verder gewerkt naar callen van single_person()

    #model_features_copy = np.array(model_features)
    model_features_copy = model_features.copy()
    # First, some safety checks ...
    # Each model must be a valid model, this means no Openpose errors (=a undetected body-part) are allowed
    #  -> models with undetected bodyparts are unvalid
    #  -> undetected body-parts are labeled by (0,0)
    if np.any(model_features_copy[:] == [0, 0]):
        logger.error(" Unvalid model pose, undetected body-parts")
        result = MatchResult(False, error_score=0, input_transformation=None)
        return result

    # Input is allowed to have a certain amount of undetected body parts
    # In that case, the corresponding point from the model is also changed to (0,0)
    #   -> afterwards matching can still proceed
    # The (0,0) points can't just be deleted because
    # because without them the featurearrays would become ambigu. (the correspondence between model and input)
    #
    # !! NOTE !! : the acceptation and introduction of (0,0) points
    # is a danger for our current normalisation
    # These particular origin points should not influence the normalisation
    # (which they do if we neglect them, xmin and ymin you know...)
    if np.any(input_features[:] == [0,0]):
        counter = 0
        for feature in input_features:
            if feature[0] == 0 and feature[1] == 0:  # (0,0)
                logger.warning(" Undetected body part in input: index(%d) %s", counter, prepocessing.get_bodypart(counter))
                model_features_copy[counter][0] = 0#np.nan
                model_features_copy[counter][1] = 0#np.nan
                input_features[counter][0] = 0#np.nan
                input_features[counter][1] = 0#np.nan
            counter = counter+1;

    assert len(model_features_copy) == len(input_features)

    # Normalise features: crop => delen door Xmax & Ymax (NIEUWE MANIER!!)
    # !Note!: as state above, care should be taken when dealing
    #   with (0,0) points during normalisation
    #
    # TODO:
    # !Note2!: The exclusion of a feature in the torso-regio doesn't effect
    #   the affine transformation in the legs- and face-regio in general.
    #   BUT in some case it CAN influence the (max-)euclidean distance.
    #     -> (so could resolve in different MATCH result)
    #   This is the case when the undetected bodypart [=(0,0)] would be the
    #   minX or minY in the detected case.
    #   Now, in the absence of this minX or minY, another feature will deliver
    #   this value.
    #   -> The normalisation region is smaller and gives different values after normalisation.
    #
    #   (BV: als iemand met handen in zij staat maar de rechter ellenboog niet gedetect wordt
    #       => minX is nu van het rechthand dat in de zij staat.

    # TODO
    # It seems like the number of excluded features is proportional with the rotation angle
    # -> That is, the more features are missing, the higher the rotation angle becomes, this is weird

    if (normalise):
        model_features_copy = normalising.feature_scaling(model_features_copy)
        input_features = normalising.feature_scaling(input_features)

    #Split features in three parts
    (model_face, model_torso, model_legs) = prepocessing.split_in_face_legs_torso(model_features_copy)
    (input_face, input_torso, input_legs) = prepocessing.split_in_face_legs_torso(input_features)

    # Zoek transformatie om input af te beelden op model
    # Returnt transformatie matrix + afbeelding/image van input op model
    (input_transformed_face, transformation_matrix_face) = affine_transformation.find_transformation(model_face, input_face)
    (input_transformed_torso, transformation_matrix_torso) = affine_transformation.find_transformation(model_torso, input_torso)
    (input_transformed_legs, transformation_matrix_legs) = affine_transformation.find_transformation(model_legs, input_legs)

    # Wrapped the transformed input in one whole pose
    input_transformation = prepocessing.unsplit(input_transformed_face, input_transformed_torso, input_transformed_legs)

    # In case of no normalisation, return here (ex; plotting)
    # Without normalisation the thresholds don't say anything
    #   -> so comparison is useless
    if(not normalise):
        result = MatchResult(None, error_score=0,
                             input_transformation=input_transformation)
        return result

    max_euclidean_error_face = pose_comparison.max_euclidean_distance(model_face, input_transformed_face)
    max_euclidean_error_torso = pose_comparison.max_euclidean_distance(model_torso, input_transformed_torso)
    max_euclidean_error_legs = pose_comparison.max_euclidean_distance(model_legs, input_transformed_legs)

    max_euclidean_error_shoulders = pose_comparison.max_euclidean_distance_shoulders(model_torso, input_transformed_torso)


    ######### THE THRESHOLDS #######
    eucl_dis_tresh_torso = 0.11 #0.065  of 0.11 ??
    rotation_tresh_torso = 40
    eucl_dis_tresh_legs = 0.055
    rotation_tresh_legs = 40

    eucld_dis_shoulders_tresh = 0.063
    ################################

    result_torso = pose_comparison.decide_torso_shoulders_incl(max_euclidean_error_torso, transformation_matrix_torso,
                                                eucl_dis_tresh_torso, rotation_tresh_torso,
                                                max_euclidean_error_shoulders, eucld_dis_shoulders_tresh)

    result_legs = pose_comparison.decide_legs(max_euclidean_error_legs, transformation_matrix_legs,
                                              eucl_dis_tresh_legs, rotation_tresh_legs)

    #TODO: construct a solid score algorithm
    error_score = (max_euclidean_error_torso + max_euclidean_error_legs)/2.0

    result = MatchResult((result_torso and result_legs), error_score=error_score, input_transformation=input_transformation)
    return result


#Plot the calculated transformation on the model image
#And some other usefull plots for debugging
#NO NORMALIZING IS DONE HERE BECAUSE POINTS ARE PLOTTED ON THE ORIGINAL PICTURES!
def plot_single_person(model_features, input_features, model_image_name, input_image_name):
    # First, some safety checks ...
    # Each model must be a valid model, this means no Openpose errors (=a undetected body-part) are allowed
    #  -> models with undetected bodyparts are unvalid
    #  -> undetected body-parts are labeled by (0,0)
    if np.any(model_features[:] == [0, 0]):
        logger.error(" Unvalid model pose, undetected body-parts")
        result = MatchResult(False, error_score=0, input_transformation=None)
        return result

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
    #ax1.set_title(model_image_name + ' (model)')
    ax1.set_title('(model)')
    ax1.plot(*zip(*model_features), marker='o', color='magenta', ls='', label='model', ms=markersize)  # ms = markersize
    red_patch = mpatches.Patch(color='red', label='model')
    #ax1.legend(handles=[red_patch])

    #ax2.set_title(input_image_name + ' (input)')
    ax2.set_title('(input)')
    ax2.imshow(input_image)
    ax2.plot(*zip(*input_features), marker='o', color='r', ls='', ms=markersize)
    #ax2.legend(handles=[mpatches.Patch(color='blue', label='input')])

    whole_input_transform = prepocessing.unsplit(input_transformed_face, input_transformed_torso, input_transformed_legs)
    ax3.set_title('Transformation of input + model')
    ax3.imshow(model_image)
    ax3.plot(*zip(*model_features), marker='o', color='magenta', ls='', label='model', ms=markersize)  # ms = markersize
    ax3.plot(*zip(*whole_input_transform), marker='o', color='b', ls='', ms=markersize)
    #ax3.legend(handles=[mpatches.Patch(color='yellow', label='Transformation of model')])

    plt.show()


'''
Description multi_person():
This function is used in the first (simple) case: MODELS SEPARATELY (mutual orientation of different poses is not checked in this simple case)
    The human poses in the image have no relation with each other and they are considered separately 
    Foreach modelpose (in models_poses) a matching inputpose (in input_poses) is searched
    Only if for each modelpose matches with on of the inputposes in input_poses, a global match is achieved. 

Parameters:
@:param models_poses: Takes an array of models as input because every pose that needs to be mimic has it's own model
@:param input_poses: The input is one json file. This represents an image of multiple persons and they each try to mimic one of the poses in model

Returns:
@:returns False : in case GLOBAL MATCH FAILED
@:returns list_of_all_matches : (List of MatchCombo objects) each model 1 match with a input (this match is wrapped in a MatchCombo object)
'''
# THE NEW one: for every modelpose , a matching input is seeked
# Enkel zo kan je een GLOBAL MATCH FAILED besluiten na dat er geen matching inputpose is gevonden voor een modelpose
def multi_person(models_poses, input_poses, model_image_name, input_image_name):
    logger.info(" Multi-person matching...")
    logger.info(" amount of models: %d", len(models_poses))
    logger.info(" amount of inputs: %d", len(input_poses))
    # Some safety checks first ..
    if(len(input_poses)== 0 or len(models_poses) == 0):
        logger.error(" Multi person match failed. Inputposes or modelposes are empty")
        return False

    # Then check if there are equal or more input poses than model poses
    # When the amount of input poses is smaller than model poses a global match FAILED is decided
    # Note: amount of models may be smaller than amount of input models:
    #       -> ex in case of people on the background which are also detected by openpose => background noise
    #       -> TODO: ? foreground detection necessary (in case less models than there are inputs)  NOT SURE ...
    if(len(input_poses) < len(models_poses)):
        logger.error(" Multi person match failed. Amount of input poses < model poses")
        return False

    # When there are more inputposes than modelsposes, print warning
    # I this case pose match still needs to proceed,
    #  BUT it's possible that a background pose is matched with one of the proposed modelposes!! (something we don't want)
    if (len(input_poses) > len(models_poses)):
        logger.warning(" !! WARNING !! Amount of input poses > model poses")
        #Continiue
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

    # Iterate over the model poses
    # TODO: improve search algorithm (not necessary i guess, as it is only illustrative)
    counter_model_pose = 1
    logger.debug(" ->Searching a best-match for each model in the modelposes ...")
    for model_pose in models_poses:
        logger.debug(" Iterate for modelpose(%d)", counter_model_pose)
        counter_input_pose = 1
        for input_pose in input_poses:
            logger.debug(" @@@@ Matching model(%d) with input(%d) @@@@", counter_model_pose, counter_input_pose)
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
        # This can mean only one thing:
        #   1. The user(s) failed to mimic one of the proposed model poses

        if(best_match_combo  is None):
            logger.info(" MATCH FAILED. No match found for modelpose(%d). User failed to match a modelpose ", counter_model_pose)
            return False

        # After comparing every possible models with a inputpose, append to match_list
        list_of_all_matches.append(best_match_combo)
        # And reset best_match field for next iteration
        best_match_combo = None

        counter_model_pose = counter_model_pose + 1


    # Plotjes: affine transformation is calculated again but now without normalisation
    for i in list_of_all_matches:
        if i is not None:
            logger.info("-- Looping over best-matches for producing plotjes:")
            (result, error_score, input_transformation) = single_person(i.model_features, i.input_features, False)
            plot_match(i.model_features, i.input_features, input_transformation, model_image_name, input_image_name)

    return list_of_all_matches


'''
Description multi_person2()
This function is used in the second (complex) case: The models are dependent of each other in space
Their relation in space is checked in the same way as in case of single_pose(), 
    but now a affine transformation of the total of all poses is calculated

First a multi_pose() is executed and a list of best_matches is achieved
Then all separate input poses are combined into one input_pose_transformed
    This is the homography of all model poses displayed onto their best match inputpose.
    -> The modelpose is superimposed onto his matching inputpose
    This homography is calculated using only a translation and rotation, NO SCALING
Note that the input_transformed resulting from single_pose() is not used in this algorithm.

Final plots are only plotted if normalised is False
# DISCLAIMER on no-normalisation: 
# It's normal that the plot is fucked up in case of undetected body parts in the input
#  -> this is because no normalisation is done here (because of the plots)
#     and thus these sneaky (0,0) points are not handled.
# TODO: maybe not include the (0,0) handler only the normalising part??

A word on input poses with undetected body parts [ (0,0) points ]:
    Input poses with a certain amount of undetected body parts are allowed. 
    It is even so that,  if a best match is found for a model, 
    in the second step (procrustes) the undetected body parts 
    are overwritten with those of the model. 

Parameters:
@:param model_poses: Model containing multiple modelposes (one json file = one image because poses are seen as one whole) 
@:param input_poses: The input is one json file. This represents an image of multiple persons and 
                        together they try to mimic the whole model. 
@:param normalise: Default is True. In case of False; the max euclidean distance is calculated and reported
                    In case of True; the result in plotted on the images! 


Returns:
@:returns False : in case GLOBAL MATCH FAILED
@:returns True : Match! 
'''
#TODO fine-tune returns
def multi_person2(model_poses, input_poses, model_image_name, input_image_name, normalise=True):
    # Find for each model_pose the best match input_pose
    # returns a list of best matches
    # TODO fine-tune return tuple
    result = multi_person(model_poses, input_poses, model_image_name, input_image_name)

    if(result is False):
        # Minimum one model pose is not matched with a input pose
        logger.error("Multi-person step1 match failed!")
        return False

    # The new input_transformed; contains all poses and wrapped in one total pose.
    # This input_transformed_combined is achieved by superimposing all the model poses on their corresponding inputpose
    input_transformed_combined = []

    # Loop over the best-matches
    #       [modelpose 1 -> inputpose x ; modelpose2 -> inputpose y; ...]
    for best_match in result:
        #Note: the input_transformed from single_pose() is not used!!!
        input_transformed = proc_do_it.superimpose(best_match.input_features, best_match.model_features, input_image_name, model_image_name)
        input_transformed_combined.append(np.array(input_transformed))

    #logger.info("size input: " + str(len(input_transformed_combined)))
    #logger.info("size model: " + str(len(model_poses)))
    assert len(input_transformed_combined) == len(model_poses)

    # TODO: harded code indexen weg doen
    # TODO: transpose van ne lijst? Mss beter toch met np.array() werken..  maar hoe init'en?
    assert len(input_transformed_combined) >= 2

    #Lijst vervormen naar matrix
    input_transformed_combined = np.vstack([input_transformed_combined[0], input_transformed_combined[1]])
    model_poses = np.vstack([model_poses[0], model_poses[1]]) #TODO waaarom moet da hier getransposed worden LOL ? zit ergens foutje in preprocessing
    input_poses = np.vstack([input_poses[0], input_poses[1]])

    # Pad with ones so our affine transformation can also do translations
    input_transformed_combined = prepocessing.pad(input_transformed_combined)

    if(normalise):
        input_transformed_combined = normalising.feature_scaling(input_transformed_combined)
        model_poses = normalising.feature_scaling(model_poses)

    # Calc the affine trans of the whole
    (full_transformation, A_matrix) = affine_transformation.find_transformation(model_poses, input_transformed_combined)

    # TODO return True in case of match
    if(normalise):
        max_eucl_distance = pose_comparison.max_euclidean_distance(model_poses, input_transformed_combined)
        logger.info("--->Max eucl distance: %s  (thresh ca. 0.13)", str(max_eucl_distance)) # torso thresh is 0.11
    else:
        plot_multi_pose(model_poses, input_poses, full_transformation,
                        model_image_name, input_image_name, "input poses", "full procrustes")

        plot_multi_pose(model_poses, input_transformed_combined, full_transformation,
                        model_image_name, input_image_name, "superimposed model on input", "full procrustes")

    return True

#Plots all Three: model, input and transformation
def plot_multi_pose(model_features, input_features, full_transform, model_image_name, input_image_name, text_input, text_transform):
    # plot vars
    markersize = 2

    # Load images
    model_image = plt.imread(model_image_name)
    input_image = plt.imread(input_image_name)

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
    implot = ax1.imshow(model_image)
    ax1.set_title('(model)')
    ax1.plot(*zip(*model_features), marker='o', color='magenta', ls='', label='model', ms=markersize)  # ms = markersize
    red_patch = mpatches.Patch(color='magenta', label='model')
    #ax1.legend(handles=[red_patch])

    ax2.set_title('('+text_input+')')
    ax2.imshow(input_image)
    ax2.plot(*zip(*input_features), marker='o', color='red', ls='', ms=markersize)
    #ax2.legend(handles=[mpatches.Patch(color='blue', label='input')])

    ax3.set_title('('+text_transform+')')
    ax3.imshow(model_image)
    ax3.plot(*zip(*model_features), marker='o', color='magenta', ls='', label='model', ms=markersize)  # ms = markersize
    ax3.plot(*zip(*full_transform), marker='o', color='blue', ls='', ms=markersize)
    ax3.legend(handles=[mpatches.Patch(color='magenta', label='Model'), mpatches.Patch(color='blue', label='Input transformed')])

    plt.show()

    return

#Plots all Three: model, input and transformation
def plot_match(model_features, input_features, input_transform_features, model_image_name, input_image_name):
    # plot vars
    markersize = 2

    # Load images
    model_image = plt.imread(model_image_name)
    input_image = plt.imread(input_image_name)

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
    implot = ax1.imshow(model_image)
    ax1.set_title('(model)')
    ax1.plot(*zip(*model_features), marker='o', color='magenta', ls='', label='model', ms=markersize)  # ms = markersize
    red_patch = mpatches.Patch(color='magenta', label='model')
    #ax1.legend(handles=[red_patch])

    ax2.set_title('(input)')
    ax2.imshow(input_image)
    ax2.plot(*zip(*input_features), marker='o', color='red', ls='', ms=markersize)
    #ax2.legend(handles=[mpatches.Patch(color='blue', label='input')])

    ax3.set_title('Transformed input on model')
    ax3.imshow(model_image)
    ax3.plot(*zip(*model_features), marker='o', color='magenta', ls='', label='model', ms=markersize)  # ms = markersize
    ax3.plot(*zip(*input_transform_features), marker='o', color='blue', ls='', ms=markersize)
    ax3.legend(handles=[mpatches.Patch(color='magenta', label='Model'), mpatches.Patch(color='blue', label='Input transformed')])

    plt.show()

    return