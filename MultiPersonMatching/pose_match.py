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
import copy
import singleperson_match

logger = logging.getLogger("multiperson_match")

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
def find_best_match(models_poses, input_poses):
    logger.debug(" Multi-person matching...")
    logger.debug(" amount of models: %d", len(models_poses))
    logger.debug(" amount of inputs: %d", len(input_poses))
    # Some safety checks first ..
    if(len(input_poses)== 0 or len(models_poses) == 0):
        logger.debug(" Multi person match failed. Inputposes or modelposes are empty")
        return False

    # Then check if there are equal or more input poses than model poses
    # When the amount of input poses is smaller than model poses a global match FAILED is decided
    # Note: amount of models may be smaller than amount of input models:
    #       -> ex in case of people on the background which are also detected by openpose => background noise
    #       -> TODO: ? foreground detection necessary (in case less models than there are inputs)  NOT SURE ...
    if(len(input_poses) < len(models_poses)):
        logger.debug(" Multi person match failed. Amount of input poses < model poses")
        return False

    # When there are more inputposes than modelsposes, print warning
    # I this case pose match still needs to proceed,
    #  BUT it's possible that a background pose is matched with one of the proposed modelposes!! (something we don't want)
    if (len(input_poses) > len(models_poses)):
        logger.debug(" !! WARNING !! Amount of input poses > model poses")
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
    used_poses = []
    # Iterate over the model poses
    # TODO: improve search algorithm (not necessary i guess, as it is only illustrative)
    counter_model_pose = 1
    logger.debug(" ->Searching a best-match for each model in the modelposes ...")
    for model_pose in models_poses:
        if np.count_nonzero(model_pose)<9 or model_pose.size <9:
            counter_model_pose = counter_model_pose + 1
            logger.debug(" @@@@ bad model(%d) @@@@", counter_model_pose)
            continue

        logger.debug(" Iterate for modelpose(%d)", counter_model_pose)
        counter_input_pose = 1
        for input_pose in input_poses:
            # check if input pose has at least 8/2=4 points for transformation and that input pose isn't used twice.
            if np.count_nonzero(input_pose)<9 or input_pose.size <9 or used_poses.count(counter_input_pose) >0:
                counter_input_pose = counter_input_pose + 1
                logger.debug(" @@@@ bad input(%d) @@@@", counter_input_pose)
                continue

            logger.debug(" @@@@ Matching model(%d) with input(%d) @@@@", counter_model_pose, counter_input_pose)
            # Do single pose matching
            (result_match, error_score, input_transformation) = singleperson_match.single_person_v2(model_pose, input_pose, True)

            if (result_match):
                match_combo = MatchCombo(error_score, counter_input_pose, counter_model_pose,model_pose, input_pose, input_transformation)

                logger.debug(" Match: %s ModelPose(%d)->InputPose(%d)", result_match, counter_model_pose, counter_input_pose)

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
            logger.debug(" MATCH FAILED. No match found for modelpose(%d). User failed to match a modelpose ", counter_model_pose)
            return False

        used_poses.append(best_match_combo.input_id)
        # After comparing every possible models with a inputpose, append to match_list
        list_of_all_matches.append(best_match_combo)
        # And reset best_match field for next iteration
        best_match_combo = None

        counter_model_pose = counter_model_pose + 1
    #end for loop
    logger.debug("-- multi_pose1(): looping over best-matches for producing plotjes:")
    # Plotjes: affine transformation is calculated again but now without normalisation
    '''
    for i in list_of_all_matches:
        if i is not None:
            (result, error_score, input_transformation) = single_person(i.model_features, i.input_features, False)
            plot_match(i.model_features, i.input_features, input_transformation, model_image_name, input_image_name)
    '''

    return list_of_all_matches

'''
Description order_matches()
try to find all model poses from left to right in input poses.
'''
def order_matches(models_poses, input_poses):
    logger.debug(" Multi-person matching...")
    logger.debug(" amount of models: %d", len(models_poses))
    logger.debug(" amount of inputs: %d", len(input_poses))
    # Some safety checks first ..
    if(len(input_poses)== 0 or len(models_poses) == 0):
        logger.debug(" Multi person match failed. Inputposes or modelposes are empty")
        return False

    # Then check if there are equal or more input poses than model poses
    # When the amount of input poses is smaller than model poses a global match FAILED is decided
    # Note: amount of models may be smaller than amount of input models:
    #       -> ex in case of people on the background which are also detected by openpose => background noise
    #       -> TODO: ? foreground detection necessary (in case less models than there are inputs)  NOT SURE ...
    if(len(input_poses) < len(models_poses)):
        logger.debug(" Multi person match failed. Amount of input poses < model poses")
        return False

    # When there are more inputposes than modelsposes, print warning
    # I this case pose match still needs to proceed,
    #  BUT it's possible that a background pose is matched with one of the proposed modelposes!! (something we don't want)
    if (len(input_poses) > len(models_poses)):
        logger.debug(" !! WARNING !! Amount of input poses > model poses")
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
    used_poses = []
    # Iterate over the model poses
    # TODO: improve search algorithm (not necessary i guess, as it is only illustrative)
    counter_model_pose = 1
    logger.debug(" ->Searching a best-match for each model in the modelposes ...")
    for model_pose in models_poses:
        if np.count_nonzero(model_pose)<9 or model_pose.size <9:
            counter_model_pose = counter_model_pose + 1
            logger.debug(" @@@@ bad model(%d) @@@@", counter_model_pose)
            continue

        logger.debug(" Iterate for modelpose(%d)", counter_model_pose)
        counter_input_pose = 1
        for input_pose in input_poses:
            # check if input pose has at least 8/2=4 points for transformation and that input pose isn't used twice.
            if np.count_nonzero(input_pose)<9 or input_pose.size <9 or used_poses.count(counter_input_pose) >0:
                counter_input_pose = counter_input_pose + 1
                logger.debug(" @@@@ bad input(%d) @@@@", counter_input_pose)
                continue

            logger.debug(" @@@@ Matching model(%d) with input(%d) @@@@", counter_model_pose, counter_input_pose)
            # Do single pose matching
            (result_match, error_score, input_transformation) = singleperson_match.single_person_v2(model_pose, input_pose, True)

            if (result_match):
                match_combo = MatchCombo(error_score, counter_input_pose, counter_model_pose,model_pose, input_pose, input_transformation)

                logger.debug(" Match: %s ModelPose(%d)->InputPose(%d)", result_match, counter_model_pose, counter_input_pose)

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
            logger.debug(" MATCH FAILED. No match found for modelpose(%d). User failed to match a modelpose ", counter_model_pose)
            return False

        used_poses.append(best_match_combo.input_id)
        # After comparing every possible models with a inputpose, append to match_list
        list_of_all_matches.append(best_match_combo)
        # And reset best_match field for next iteration
        best_match_combo = None

        counter_model_pose = counter_model_pose + 1
    #end for loop
    logger.debug("-- multi_pose1(): looping over best-matches for producing plotjes:")
    # Plotjes: affine transformation is calculated again but now without normalisation
    '''
    for i in list_of_all_matches:
        if i is not None:
            (result, error_score, input_transformation) = single_person(i.model_features, i.input_features, False)
            plot_match(i.model_features, i.input_features, input_transformation, model_image_name, input_image_name)
    '''

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
#TODO optimaliseren voor geval van normalise! nu ist 2 in 1, ma voor productie is enkel normalise nodig in feite (ook ni helemaal waar -> feedback mss)
def multi_person(model_poses, input_poses, normalise=True):
    # Find for each model_pose the best match input_pose
    # returns a list of best matches !! WITH normalisation !!
    # TODO fine-tune return tuple
    result = find_best_match(model_poses, input_poses)

    if(result is False):
        # Minimum one model pose is not matched with a input pose
        logger.debug("Multi-person step1 match failed!")
        result = MatchResult(False, error_score=0, input_transformation=None)
        return result


    aantal_models = len(result)
    input_transformed_combined = np.zeros((18*aantal_models, 2))

    # The new input_transformed; contains all poses and wrapped in one total pose.
    # This input_transformed_combined is achieved by superimposing all the model poses on their corresponding inputpose
    input_transformed_combined = []

    updated_models_combined = []


    # Loop over the best-matches
    #       [modelpose 1 -> inputpose x ; modelpose2 -> inputpose y; ...]
    logger.debug("-- multi_pose(): looping over best-matches for procrustes:")

    for best_match in result:
        # First check for undetected body parts. If present=> make corresponding point in model also (0,0)
        # We can know strip them from our poses because we don't use split() for affine trans
        # TODO: deze clean updated_model_pose wordt eigenlijk al eens berekent in single_pose()
        #   -> loopke hier opnieuw is stevig redundant

        # make a array with the indecex of undetected points
        indexes_undetected_points = []
        if np.any(best_match.input_features[:] == [0, 0]):
            assert True
            counter = 0
            for feature in best_match.input_features:
                if feature[0] == 0 and feature[1] == 0:  # (0,0)
                    indexes_undetected_points.append(counter)
                    #logger.warning(" Undetected body part in input: index(%d) %s", counter,prepocessing.get_bodypart(counter))
                    best_match.model_features[counter][0] = 0
                    best_match.model_features[counter][1] = 0
                counter = counter + 1




        (input_transformed,model) = proc_do_it.superimpose(best_match.input_features, best_match.model_features)

        input_transformed_combined.append(np.array(input_transformed))
        updated_models_combined.append(np.array(model))

        #logger.info("inputtt %s", str(input_transformed))
        #logger.info("modeelll %s ", str(best_match.model_features))

    assert len(input_transformed_combined) == len(updated_models_combined)
    #not enough corresponding points
    if not (len(input_transformed_combined) >0 ):
        logger.debug("not enough corresponding points between model and input")
        result = MatchResult(False, error_score=0, input_transformation=None)
        return result

    # TODO: harded code indexen weg doen
    # TODO: transpose van ne lijst? Mss beter toch met np.array() werken..  maar hoe init'en?
    # TODO : hier is wa refactoring/optimalisatie nodig ...

    #Lijst vervormen naar matrix

    input_transformed_combined = np.vstack(input_transformed_combined)
    #model_poses = np.vstack([model_poses[0], model_poses[1]])
    model_poses =np.vstack(updated_models_combined)


    if(normalise):
        input_transformed_combined = normalising.feature_scaling(input_transformed_combined)
        model_poses = normalising.feature_scaling(model_poses)

    # Calc the affine trans of the whole
    (full_transformation, A_matrix) = affine_transformation.find_transformation(model_poses, input_transformed_combined)
    '''
    # TODO return True in case of match
    if(normalise):
        max_eucl_distance = pose_comparison.max_euclidean_distance(model_poses, input_transformed_combined)
        logger.info("--->Max eucl distance: %s  (thresh ca. 0.13)", str(max_eucl_distance)) # torso thresh is 0.11

        markersize = 2

        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
        ax1.set_title('(input transformed (model superimposed on input )')
        ax1.plot(*zip(*input_transformed_combined), marker='o', color='r', ls='', label='model', ms=markersize)  # ms = markersize

        ax2.set_title('(model)')
        ax2.plot(*zip(*model_poses), marker='o', color='r', ls='', label='model', ms=markersize)  # ms = markersize

        ax3.set_title('(affine trans and model (red))')
        ax3.plot(*zip(*full_transformation), marker='o', color='r', ls='', label='model', ms=markersize)  # ms = markersize
        ax3.plot(*zip(*model_poses), marker='o', color='b', ls='', label='model',
                 ms=markersize)  # ms = markersize
        ax = plt.gca()
        ax.invert_yaxis()
        #plt.show()
        plt.draw()


    else:
        logger.info("-- multi_pose2(): procrustes plotjes incoming ")
        plot_multi_pose(model_poses, input_poses, full_transformation,
                        model_image_name, input_image_name, "input poses", "full procrustes")
        plot_multi_pose(model_poses, input_transformed_combined, full_transformation,
                        model_image_name, input_image_name, "superimposed model on input", "full procrustes")

    #Block plots
    plt.show()
'''
    return (True,0,0)

#Plots all Three: model, input and transformation
def plot_multi_pose(model_image_name, input_image_name, list_of_all_matches):

    for i in list_of_all_matches:
        if i is not None:
            (result, error_score, input_transformation) = singleperson_match.single_person_v2(i.model_features, i.input_features, False)
            plot_match(i.model_features, i.input_features, input_transformation, model_image_name, input_image_name)

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
    plt.draw()
    #plt.show()

    return
