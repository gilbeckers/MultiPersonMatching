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


def find_best_match(models_poses, input_poses):
    logger.debug(" Multi-person matching...")
    logger.debug(" amount of models: %d", len(models_poses))
    logger.debug(" amount of inputs: %d", len(input_poses))
    # Some safety checks first ..
    if(len(input_poses)== 0 or len(models_poses) == 0):
        logger.debug(" Multi person match failed. Inputposes or modelposes are empty")
        return False


    if(len(input_poses) < len(models_poses)):
        logger.debug(" Multi person match failed. Amount of input poses < model poses")
        return False


    if (len(input_poses) > len(models_poses)):
        logger.debug(" !! WARNING !! Amount of input poses > model poses")

    list_of_all_matches = []
    model_poses = order_poses(model_poses)
    input_poses = order_poses(input_poses)
    best_match_combo = None
    used_poses = []

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

def multi_person(model_poses, input_poses, normalise=True):

    result = find_best_match(model_poses, input_poses)

    if(result is False):
        # Minimum one model pose is not matched with a input pose
        logger.debug("Multi-person step1 match failed!")
        result = MatchResult(False, error_score=0, input_transformation=None)
        return result


    aantal_models = len(result)
    input_transformed_combined = np.zeros((18*aantal_models, 2))

    input_transformed_combined = []

    updated_models_combined = []


    # Loop over the best-matches
    #       [modelpose 1 -> inputpose x ; modelpose2 -> inputpose y; ...]
    logger.debug("-- multi_pose(): looping over best-matches for procrustes:")

    for best_match in result:

        indexes_undetected_points = []
        if np.any(best_match.input_features[:] == [0, 0]):
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


    assert len(input_transformed_combined) == len(updated_models_combined)
    #not enough corresponding points
    if not (len(input_transformed_combined) >0 ):
        logger.debug("not enough corresponding points between model and input")
        result = MatchResult(False, error_score=0, input_transformation=None)
        return result

    input_transformed_combined = np.vstack(input_transformed_combined)
    model_poses =np.vstack(updated_models_combined)


    if(normalise):
        input_transformed_combined = normalising.feature_scaling(input_transformed_combined)
        model_poses = normalising.feature_scaling(model_poses)

    # Calc the affine trans of the whole
    (full_transformation, A_matrix) = affine_transformation.find_transformation(model_poses, input_transformed_combined)

    return (True,0,0)

def order_poses(poses):
    ordered = []
    for i in range(0,len(poses)):
        if(np.nonzero(poses[i]) > 8):
            pose= poses[i][:,0]
            placed = False
            place_counter = 1
            while not placed:
                place = i - place_counter
                if (place > -1):
                    prev_pose = poses[place][:,0]
                    try:
                        if np.min(pose[np.nonzero(pose)]) < np.min(prev_pose[np.nonzero(prev_pose)]):
                            place_counter = place_counter+1
                        else:
                            ordered.insert(1,poses[i])
                            placed = True
                    except ValueError:
                        placed =True
                else:
                    ordered.insert(0,poses[i])
                    placed = True
    return ordered

def find_ordered_matches(model_poses,input_poses):
    if(len(input_poses)== 0 or len(model_poses) == 0):
        logger.debug(" Multi person match failed. Inputposes or modelposes are empty")
        return False

    if(len(input_poses) < len(model_poses)):
        logger.debug(" Multi person match failed. Amount of input poses < model poses")
        return False

    if (len(input_poses) > len(model_poses)):
        logger.debug(" !! WARNING !! Amount of input poses > model poses")

    model_poses = order_poses(model_poses)
    input_poses = order_poses(input_poses)
    model_pose = model_poses[0]
    matches = []

    #find first match of poses
    for model_counter in range(0,len(model_poses)):
        model_pose = model_poses[model_counter]
        matches.append([])
        match_found = False
        start_input =0
        if model_counter >0:
            start_input =  matches[model_counter-1][0]
        for input_counter in range(0,len(input_poses)):
            input_pose = input_poses[input_counter]
            # Do single pose matching
            (result_match, error_score, input_transformation) = singleperson_match.single_person_v2(model_pose, input_pose, True)
            if result_match:
                match_found = True
                matches[model_counter].append(input_counter)
        if match_found == False:
            logger.debug("no match found for model %d", model_counter)
            return False

    logger.debug("matches found %s"," ".join(str(e) for e in matches))
    return matches

def multi_person_ordered(model_poses, input_poses, normalise=True):

    matches = find_ordered_matches(model_poses,input_poses)
    if matches == False:
        return MatchResult(False, error_score=0, input_transformation=None)
    #np = np.array(matches)
    possiblities = cartesian(matches)

    return MatchResult(True, error_score=0, input_transformation=None)


def cartesian(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def plot_multi_pose(model_image_name, input_image_name, list_of_all_matches):

    for i in list_of_all_matches:
        if i is not None:
            (result, error_score, input_transformation) = singleperson_match.single_person_v2(i.model_features, i.input_features, False)
            plot_match(i.model_features, i.input_features, input_transformation, model_image_name, input_image_name)

    return


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

    ax2.set_title('(input)')
    ax2.imshow(input_image)
    ax2.plot(*zip(*input_features), marker='o', color='red', ls='', ms=markersize)

    ax3.set_title('Transformed input on model')
    ax3.imshow(model_image)
    ax3.plot(*zip(*model_features), marker='o', color='magenta', ls='', label='model', ms=markersize)  # ms = markersize
    ax3.plot(*zip(*input_transform_features), marker='o', color='blue', ls='', ms=markersize)
    ax3.legend(handles=[mpatches.Patch(color='magenta', label='Model'), mpatches.Patch(color='blue', label='Input transformed')])
    plt.draw()
    #plt.show()

    return
