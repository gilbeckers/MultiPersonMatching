import collections
import normalising
import prepocessing
import affine_transformation
import pose_comparison
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import logging
import numpy as np
import copy

logger = logging.getLogger("singleperson_match")
MatchResult = collections.namedtuple("MatchResult", ["match_bool", "error_score", "input_transformation"])
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
@:returns model_features => is needed in multi_person2() and when (0,0) are added to modelpose
'''
def single_person(model_features, input_features, normalise=True):
    # TODO: Make a local copy ??
    # Because np.array is a mutable type => passed by reference
    #   -> dus als model wordt veranderd wordt er met gewijzigde array
    #       verder gewerkt naar callen van single_person()
    #model_features_copy = np.array(model_features)


    model_features_copy = copy.copy(model_features)

    # First, some safety checks ...
    # Each model must be a valid model, this means no Openpose errors (=a undetected body-part) are allowed
    #  -> models with undetected bodyparts are unvalid
    #  -> undetected body-parts are labeled by (0,0)
    '''
    if np.any(model_features_copy[:] == [0, 0]):
        for i in range(0,17):
            if model_features_copy[i][0] == 0 and model_features_copy[i][1] == 0:
                logger.warning(" Unvalid model pose, undetected body-parts")
                #result = MatchResult(False, error_score=0, input_transformation=None)
                #return result
    '''
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
    counter_not_found_points = 0
    if np.any(input_features[:] == [0,0]):
        counter = 0
        for feature in input_features:
            if feature[0] == 0 and feature[1] == 0:  # (0,0)
                #logger.warning(" Undetected body part in input: index(%d) %s", counter, prepocessing.get_bodypart(counter))
                if not (model_features_copy[counter][0] == 0 and model_features_copy[counter][1] == 0):
                    counter_not_found_points = counter_not_found_points+1
                model_features_copy[counter][0] = 0#np.nan
                model_features_copy[counter][1] = 0#np.nan
                #input_features[counter][0] = 0#np.nan
                #input_features[counter][1] = 0#np.nan

            counter = counter+1

    # if the input has more then 4 points not recognised then the model, then return false
    if counter_not_found_points > 2:
        logger.debug("Model has more feature then input therefore not matched")
        result = MatchResult(False,
                             error_score=0,
                             input_transformation=None)
        return result

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
        result = MatchResult(None,
                             error_score=0,
                             input_transformation=input_transformation)
        return result

    max_euclidean_error_face = pose_comparison.max_euclidean_distance(model_face, input_transformed_face)
    max_euclidean_error_torso = pose_comparison.max_euclidean_distance(model_torso, input_transformed_torso)
    max_euclidean_error_legs = pose_comparison.max_euclidean_distance(model_legs, input_transformed_legs)

    max_euclidean_error_shoulders = pose_comparison.max_euclidean_distance_shoulders(model_torso, input_transformed_torso)


    ######### THE THRESHOLDS #######
    eucl_dis_tresh_torso = 0.068 #0.065  of 0.11 ??
    rotation_tresh_torso = 13.606
    eucl_dis_tresh_legs = 0.045
    rotation_tresh_legs = 14.677
    eucld_dis_shoulders_tresh = 0.068
    ################################

    result_torso = pose_comparison.decide_torso_shoulders_incl(max_euclidean_error_torso, transformation_matrix_torso,
                                                eucl_dis_tresh_torso, rotation_tresh_torso,
                                                max_euclidean_error_shoulders, eucld_dis_shoulders_tresh)

    result_legs = pose_comparison.decide_legs(max_euclidean_error_legs, transformation_matrix_legs,
                                              eucl_dis_tresh_legs, rotation_tresh_legs)

    #TODO: construct a solid score algorithm
    error_score = (max_euclidean_error_torso + max_euclidean_error_legs)/2.0

    result = MatchResult((result_torso and result_legs),
                         error_score=error_score,
                         input_transformation=input_transformation)
    return result

def single_person_v2(model_features, input_features, normalise=True):
    # TODO: Make a local copy ??
    # Because np.array is a mutable type => passed by reference
    #   -> dus als model wordt veranderd wordt er met gewijzigde array
    #       verder gewerkt naar callen van single_person()
    #model_features_copy = np.array(model_features)

    model_features_copy = copy.copy(model_features)

    # First, some safety checks ...
    # Each model must be a valid model, this means no Openpose errors (=a undetected body-part) are allowed
    #  -> models with undetected bodyparts are unvalid
    #  -> undetected body-parts are labeled by (0,0)
    '''
    if np.any(model_features_copy[:] == [0, 0]):
        for i in range(0,17):
            if model_features_copy[i][0] == 0 and model_features_copy[i][1] == 0:
                logger.warning(" Unvalid model pose, undetected body-parts")
                #result = MatchResult(False, error_score=0, input_transformation=None)
                #return result
    '''
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
    counter_not_found_points = 0
    if np.any(input_features[:] == [0,0]):
        counter = 0
        for feature in input_features:
            if feature[0] == 0 and feature[1] == 0:  # (0,0)
                #logger.warning(" Undetected body part in input: index(%d) %s", counter, prepocessing.get_bodypart(counter))
                if not (model_features_copy[counter][0] == 0 and model_features_copy[counter][1] == 0):
                    counter_not_found_points = counter_not_found_points+1
                model_features_copy[counter][0] = 0#np.nan
                model_features_copy[counter][1] = 0#np.nan
                #input_features[counter][0] = 0#np.nan
                #input_features[counter][1] = 0#np.nan

            counter = counter+1

    # if the input has more then 4 points not recognised then the model, then return false
    if counter_not_found_points > 2:
        logger.debug("Model has more feature then input therefore not matched")
        result = MatchResult(False,
                             error_score=0,
                             input_transformation=None)
        return result

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
            # In case of no normalisation, return here (ex; plotting)
            # Without normalisation the thresholds don't say anything
            #   -> so comparison is useless
    else:
        result = MatchResult(None,
                             error_score=0,
                             input_transformation=input_transformation)
        return result

    #Split features in three parts
    (model_face, model_torso, model_legs) = prepocessing.split_in_face_legs_torso_v2(model_features_copy)
    (input_face, input_torso, input_legs) = prepocessing.split_in_face_legs_torso_v2(input_features)

    ######### THE THRESHOLDS #######
    eucl_dis_tresh_torso = 0.098
    rotation_tresh_torso = 10.847
    eucl_dis_tresh_legs = 0.05
    rotation_tresh_legs = 14.527
    eucld_dis_shoulders_tresh = 0.085
    ################################
    #handle face
    (input_transformed_face, transformation_matrix_face) = affine_transformation.find_transformation(model_face, input_face)
    max_euclidean_error_face = pose_comparison.max_euclidean_distance(model_face, input_transformed_face)
    if  np.count_nonzero(model_face)>5:

        #
        result_face = True
    else:
        result_face = True

    #handle Torso
    (input_transformed_torso, transformation_matrix_torso) = affine_transformation.find_transformation(model_torso, input_torso)
    max_euclidean_error_torso = pose_comparison.max_euclidean_distance(model_torso, input_transformed_torso)
    max_euclidean_error_shoulders = pose_comparison.max_euclidean_distance_shoulders(model_torso, input_transformed_torso)
    if  np.count_nonzero(model_torso)>5:
        result_torso = pose_comparison.decide_torso_shoulders_incl(max_euclidean_error_torso, transformation_matrix_torso,
                                                    eucl_dis_tresh_torso, rotation_tresh_torso,
                                                    max_euclidean_error_shoulders, eucld_dis_shoulders_tresh)
    else:
        result_torso = True

    #handle legs
    (input_transformed_legs, transformation_matrix_legs) = affine_transformation.find_transformation(model_legs, input_legs)
    max_euclidean_error_legs = pose_comparison.max_euclidean_distance(model_legs, input_transformed_legs)
    if  np.count_nonzero(model_legs)>5:
        result_legs = pose_comparison.decide_legs(max_euclidean_error_legs, transformation_matrix_legs,
                                                      eucl_dis_tresh_legs, rotation_tresh_legs)
    else:
        result_legs = True

    # Wrapped the transformed input in one whole pose
    input_transformation = prepocessing.unsplit(input_transformed_face, input_transformed_torso, input_transformed_legs)


    #TODO: construct a solid score algorithm
    error_score = (max_euclidean_error_torso + max_euclidean_error_legs)/2.0

    result = MatchResult((result_torso and result_legs and result_face),
                         error_score=error_score,
                         input_transformation=input_transformation)
    return result


def single_person_treshholds(model_features, input_features,eucl_dis_tresh_torso,rotation_tresh_torso,eucl_dis_tresh_legs,rotation_tresh_legs,eucld_dis_shoulders_tresh, normalise=True):
    # TODO: Make a local copy ??
    # Because np.array is a mutable type => passed by reference
    #   -> dus als model wordt veranderd wordt er met gewijzigde array
    #       verder gewerkt naar callen van single_person()
    #model_features_copy = np.array(model_features)

    model_features_copy = model_features.copy()

    # First, some safety checks ...
    # Each model must be a valid model, this means no Openpose errors (=a undetected body-part) are allowed
    #  -> models with undetected bodyparts are unvalid
    #  -> undetected body-parts are labeled by (0,0)
    if np.any(model_features_copy[:] == [0, 0]):
        for i in range(0,17):
            if model_features_copy[i][0] == 0 and model_features_copy[i][1] == 0:
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
                #logger.warning(" Undetected body part in input: index(%d) %s", counter, prepocessing.get_bodypart(counter))
                model_features_copy[counter][0] = 0#np.nan
                model_features_copy[counter][1] = 0#np.nan
                #input_features[counter][0] = 0#np.nan
                #input_features[counter][1] = 0#np.nan
            counter = counter+1

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
    #   this value.model_features_copy = model_features.copy()

    # First, some safety checks ...
    # Each model must be a valid model, this means no Openpose errors (=a undetected body-part) are allowed
    #  -> models with undetected bodyparts are unvalid
    #  -> undetected body-parts are labeled by (0,0)
    '''
    if np.any(model_features_copy[:] == [0, 0]):
        for i in range(0,17):
            if model_features_copy[i][0] == 0 and model_features_copy[i][1] == 0:
                logger.warning(" Unvalid model pose, undetected body-parts")
                #result = MatchResult(False, error_score=0, input_transformation=None)
                #return result
    '''
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
                #logger.warning(" Undetected body part in input: index(%d) %s", counter, prepocessing.get_bodypart(counter))
                model_features_copy[counter][0] = 0#np.nan
                model_features_copy[counter][1] = 0#np.nan
                #input_features[counter][0] = 0#np.nan
                #input_features[counter][1] = 0#np.nan
            counter = counter+1

    assert len(model_features_copy) == len(input_features)
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
        result = MatchResult(None,
                             error_score=0,
                             input_transformation=input_transformation)
        return result

    max_euclidean_error_face = pose_comparison.max_euclidean_distance(model_face, input_transformed_face)
    max_euclidean_error_torso = pose_comparison.max_euclidean_distance(model_torso, input_transformed_torso)
    max_euclidean_error_legs = pose_comparison.max_euclidean_distance(model_legs, input_transformed_legs)

    max_euclidean_error_shoulders = pose_comparison.max_euclidean_distance_shoulders(model_torso, input_transformed_torso)

    ######### THE THRESHOLDS #######
    '''
    eucl_dis_tresh_torso = 0.11 #0.065  of 0.11 ??
    rotation_tresh_torso = 40
    eucl_dis_tresh_legs = 0.055
    rotation_tresh_legs = 40

    eucld_dis_shoulders_tresh = 0.063
    '''
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
    # First, some safety checks ...
    # Each model must be a valid model, this means no Openpose errors (=a undetected body-part) are allowed
    #  -> models with undetected bodyparts are unvalid
    #  -> undetected body-parts are labeled by (0,0)

    if np.any(model_features[:] == [0, 0]):
        for i in range(0,17):
            if model_features[i][0] == 0 and model_features[i][1] == 0:
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
