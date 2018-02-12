import numpy as np
import logging

logger = logging.getLogger("pose_match")

def split_in_face_legs_torso(features):
    # torso = features[2:8]   #zonder nek
    torso = features[1:8]   #met nek  => if nek incl => compare_incl_schouders aanpassen!!
    legs = features[8:14]
    face = np.vstack([features[0], features[14:18]])

    return (face, torso, legs)

def unsplit(face, torso, legs):
    whole = np.vstack([face[0], torso, legs, face[1:5]])

    return whole

def handle_undetected_points(input_features, model_features):
    # Because np.array is a mutable type => passed by reference
    #   -> dus als model wordt veranderd wordt er met gewijzigde array
    #       verder gewerkt na callen van single_person()
    # model_features_copy = np.array(model_features)
    model_features_copy = model_features.copy()
    input_features_copy = input_features.copy()

    # Input is allowed to have a certain amount of undetected body parts
    # In that case, the corresponding point from the model is also changed to (0,0)
    #   -> afterwards matching can still proceed
    # The (0,0) points can't just be deleted because
    # without them the feature-arrays would become ambigu. (the correspondence between model and input)
    #
    # !! NOTE !! : the acceptation and introduction of (0,0) points
    # is a danger for our current normalisation
    # These particular origin points should not influence the normalisation
    # (which they do if we neglect them, xmin and ymin you know ... )
    if np.any(input_features[:] == [0, 0]):
        counter = 0
        for feature in input_features:
            if feature[0] == 0 and feature[1] == 0:  # (0,0)
                logger.warning(" Undetected body part in input: index(%d) %s", counter,
                               get_bodypart(counter))
                model_features_copy[counter][0] = 0
                model_features_copy[counter][1] = 0
                # input_features[counter][0] = 0#np.nan
                # input_features[counter][1] = 0#np.nan
            counter = counter + 1

    # In this second version, the model is allowed to have undetected features
    if np.any(model_features[:] == [0, 0]):
        counter = 0
        for feature in model_features:
            if feature[0] == 0 and feature[1] == 0:  # (0,0)
                logger.warning(" Undetected body part in MODEL: index(%d) %s", counter,
                               get_bodypart(counter))
                input_features_copy[counter][0] = 0
                input_features_copy[counter][1] = 0
            counter = counter + 1

    assert len(model_features_copy) == len(input_features_copy)

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
    # -> NIET ECHT RAAR EIGENLIJK WANT MINDER punten betekent minder constraints, waardoor er meer kan gedraaid worden (meer vrijheidsgraad)

    return (input_features_copy, model_features_copy)



def unpad(matrix):

    return matrix[:, :-1]

def pad(matrix):
    return np.hstack([matrix, np.zeros((matrix.shape[0], 1))])

import numpy as np

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

options = {0 : 'neus',
           1 : 'nek',
           2 : 'l-schouder',
           3 : 'l-elleboog',
           4 : 'l-pols',
           5 : 'r-schouder',
           6 : 'r-elleboog',
           7 : 'r-pols',
           8 : 'l-heup',
           9 : 'l-knie',
           10: 'l-enkel',
           11: 'r-heup',
           12: 'r-knie',
           13: 'r-enkel',
           14: 'l-oog',
           15: 'r-oog',
           16: 'l-oor',
           17: 'r-oor',
        }

def get_bodypart(index):

    if(index <=17 and index >=0):
        return options[index]

    return 'no-bodypart (wrong index)'


def corr2_coeff(A,B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);

    # Finally get corr coeff
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))
