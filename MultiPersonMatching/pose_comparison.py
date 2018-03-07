import numpy as np
import logging
logger = logging.getLogger("pose_match")


#Beslist op basis van max_euclidean distance en rotatie hoek en !!! aparte threshold op euclid distance van schouders !!!!
#Geeft final boolean terug= heel part (torso, legs of face) is match of geen match
def decide_torso_shoulders_incl(max_euclid_distance_torso, transformation_matrix, eucld_tresh, rotation_tresh,
                                max_euclid_distance_shoulders, shoulder_thresh):
        if not len(transformation_matrix) ==0:
        #Calcuation rotation of transformation
            rotation_1 = np.abs(np.math.atan2(-transformation_matrix[0][1], transformation_matrix[0][0]) * 57.3)
            rotation_2 = np.abs(np.math.atan2(transformation_matrix[1][0], transformation_matrix[1][1]) * 57.3)
            rot_max = max(rotation_2, rotation_1)
        else:
            rot_max =0
        logger.debug(" --- Evaluate Torso---")
        logger.debug(" max eucldis: %s  thresh(%s)", max_euclid_distance_torso, eucld_tresh)
        logger.debug(" max rot:     %s  thresh(%s)", rot_max, rotation_tresh)
        logger.debug(" max shoulder:%s  thresh(%s)", max_euclid_distance_shoulders, shoulder_thresh)

        # Zeker juist, dus match
        if (max_euclid_distance_torso <= eucld_tresh and rot_max <= rotation_tresh):

            # Checken of schouders niet te veel afwijken
            if (max_euclid_distance_shoulders <= shoulder_thresh):
                logger.debug("\t ->#TORSO MATCH#")
                return True
            else:
                logger.debug("!!!!!TORSO NO MATCH Schouder error te groot!!!!")

        # Geen match
        logger.debug("\t ->#TORSO NO MATCH#")
        return False


#Evaluate legs ..
def decide_legs(max_error, transformation_matrix, eucld_tresh, rotation_tresh):
    if not len(transformation_matrix) ==0:
        rotation_1 = np.abs(np.math.atan2(-transformation_matrix[0][1], transformation_matrix[0][0]) * 57.3)
        rotation_2 = np.abs(np.math.atan2(transformation_matrix[1][0], transformation_matrix[1][1]) * 57.3)
        rot_max = max(rotation_2, rotation_1)
    else:
        rot_max =0


    logger.debug(" --- Evaluate Legs---")
    logger.debug(" max eucldis: %s thresh(%s)", max_error, eucld_tresh)
    logger.debug(" max rot:     %s thresh(%s)", rot_max, rotation_tresh)

    # Zeker juist, dus match
    if (max_error <= eucld_tresh and rot_max <= rotation_tresh):
        logger.debug("\t ->#LEGS MATCH#")
        return True

    #Geen match
    logger.debug("\t ->#LEGS NO-MATCH#")
    return False


def max_euclidean_distance_shoulders(model_torso, input_transformed_torso):
    maxError_torso = np.abs(model_torso - input_transformed_torso)

    euclDis_torso = ((maxError_torso[:, 0]) ** 2 + maxError_torso[:, 1] ** 2) ** 0.5

    # Opgelet!! als nek er niet in zit is linker schouder = index 0 en rechterschouder = index 3
    # indien nek incl = > index 1 en index 4
    maxError_shoulder = max([euclDis_torso[1], euclDis_torso[4]])
    return maxError_shoulder

def max_euclidean_distance(model, transformed_input):

    manhattan_distance = np.abs(model - transformed_input)

    euclidean_distance = ((manhattan_distance[:, 0]) ** 2 + manhattan_distance[:, 1] ** 2) ** 0.5

    return max(euclidean_distance)
