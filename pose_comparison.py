import numpy as np
import config


#Beslist op basis van max_euclidean distance en rotatie hoek en !!! aparte threshold op euclid distance van schouders !!!!
#Geeft final boolean terug= heel part (torso, legs of face) is match of geen match
def decide_torso_shoulders_incl(max_euclid_distance_torso, transformation_matrix, eucld_tresh, rotation_tresh,
                                max_euclid_distance_shoulders, shoulder_thresh):

        #Calcuation rotation of transformation
        rotation_1 = np.abs(np.math.atan2(-transformation_matrix[0][1], transformation_matrix[0][0]) * 57.3)
        rotation_2 = np.abs(np.math.atan2(transformation_matrix[1][0], transformation_matrix[1][1]) * 57.3)
        rot_max = max(rotation_2, rotation_1)

        if(config.DEBUG_PRINT):
            print("--- Evaluate Torso---")
            print("max eucldis: ", max_euclid_distance_torso)
            print("max rot:     ", rot_max)
            print("max shoulder: ", max_euclid_distance_shoulders)

        # Zeker juist, dus match
        if (max_euclid_distance_torso <= eucld_tresh and rot_max <= rotation_tresh):

            # Checken of schouders niet te veel afwijken
            if (max_euclid_distance_shoulders < shoulder_thresh):
                if(config.DEBUG_PRINT):
                    print("#TORSO MATCH#")
                return True
            else:
                if(config.DEBUG_PRINT):
                    print("!!!!!TORO NO MATCH Schouder error te groot!!!!")

        # Geen match
        if (config.DEBUG_PRINT):
            print("TORO NO MATCH ")
        return False


def decide_torso_shoulders_incl(max_euclid_distance_torso, transformation_matrix, eucld_tresh, rotation_tresh,
                                max_euclid_distance_shoulders, shoulder_thresh):
    # Calcuation rotation of transformation
    rotation_1 = np.abs(np.math.atan2(-transformation_matrix[0][1], transformation_matrix[0][0]) * 57.3)
    rotation_2 = np.abs(np.math.atan2(transformation_matrix[1][0], transformation_matrix[1][1]) * 57.3)
    rot_max = max(rotation_2, rotation_1)

    if (config.DEBUG_PRINT):
        print("--- Evaluate Torso---")
        print("max eucldis: ", max_euclid_distance_torso)
        print("max rot:     ", rot_max)
        print("max shoulder: ", max_euclid_distance_shoulders)

    # Zeker juist, dus match
    if (max_euclid_distance_torso <= eucld_tresh and rot_max <= rotation_tresh):

        # Checken of schouders niet te veel afwijken
        if (max_euclid_distance_shoulders < shoulder_thresh):
            if (config.DEBUG_PRINT):
                print("\t#TORSO MATCH#")
            return True
        else:
            if (config.DEBUG_PRINT):
                print("!!!!!TORO NO MATCH Schouder error te groot!!!!")

    # Geen match
    if (config.DEBUG_PRINT):
        print("\tTORO NO MATCH ")
    return False

def decide_legs(max_error, transformation_matrix, eucld_tresh, rotation_tresh):
    rotation_1 = np.abs(np.math.atan2(-transformation_matrix[0][1], transformation_matrix[0][0]) * 57.3)
    rotation_2 = np.abs(np.math.atan2(transformation_matrix[1][0], transformation_matrix[1][1]) * 57.3)
    rot_max = max(rotation_2, rotation_1)

    if (config.DEBUG_PRINT):
        print("\n--- Evaluate Torso---")
        print("max eucldis: ", max_error)
        print("max rot:     ", rot_max)

    # Zeker juist, dus match
    if (max_error <= eucld_tresh and rot_max <= rotation_tresh):
        if (config.DEBUG_PRINT):
            print("\t#LEGS MATCH#")
        return True

    #Geen match
    if (config.DEBUG_PRINT):
        print("\t#LEGS NO MATCH#")
    return False


def max_euclidean_distance_shoulders(model_torso, input_transformed_torso):
    maxError_torso = np.abs(model_torso - input_transformed_torso)

    euclDis_torso = ((maxError_torso[:, 0]) ** 2 + maxError_torso[:, 1] ** 2) ** 0.5

    maxError_shoulder = max([euclDis_torso[0], euclDis_torso[3]])
    return maxError_shoulder

def max_euclidean_distance(model, transformed_input):

    manhattan_distance = np.abs(model - transformed_input)

    euclidean_distance = ((manhattan_distance[:, 0]) ** 2 + manhattan_distance[:, 1] ** 2) ** 0.5

    return max(euclidean_distance)
