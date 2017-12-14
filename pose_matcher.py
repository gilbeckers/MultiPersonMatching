import normalising
import numpy as np
import calcTransformationMatrix

def is_match_legs (model, input,euclDis_tresh_legs):
    primary_legs = model[8:14]
    secondary_legs = input[8:14]
    (modelTransform_legs, A_legs) = calcTransformationMatrix.calcTransformationMatrix(primary_legs, secondary_legs)
    maxError_legs = np.abs(secondary_legs - modelTransform_legs)
    euclDis_legs = ((maxError_legs[:, 0]) ** 2 + maxError_legs[:, 1] ** 2) ** 0.5
    max_euclDis_legs = max(euclDis_legs)

    return max_euclDis_legs <= euclDis_tresh_legs

def is_match_torso (model, input,euclDis_tresh_torso):
    primary_torso = model[2:8]
    secondary_torso = input[2:8]
    (modelTransform_torso, A_torso) = calcTransformationMatrix.calcTransformationMatrix(primary_torso, secondary_torso)
    maxError_torso = np.abs(secondary_torso - modelTransform_torso)
    euclDis_torso = ((maxError_torso[:, 0]) ** 2 + maxError_torso[:, 1] ** 2) ** 0.5
    max_euclDis_torso = max(euclDis_torso)

    return max_euclDis_torso <= euclDis_tresh_torso


def is_match_shoulder (model, input,tresh_shoulder):
    primary_torso = model[2:8]
    secondary_torso = input[2:8]
    (modelTransform_torso, A_torso) = calcTransformationMatrix.calcTransformationMatrix(primary_torso, secondary_torso)
    maxError_torso = np.abs(secondary_torso - modelTransform_torso)
    euclDis_torso = ((maxError_torso[:, 0]) ** 2 + maxError_torso[:, 1] ** 2) ** 0.5
    max_shoulder = max([euclDis_torso[0],euclDis_torso[3]])
    return max_shoulder<tresh_shoulder

def is_match_rotation_legs(model, input,rotation_tresh):
    primary_legs = model[8:14]
    secondary_legs = input[8:14]
    (modelTransform_legs, A_legs) = calcTransformationMatrix.calcTransformationMatrix(primary_legs, secondary_legs)
    rotation_1 = np.abs(np.math.atan2(-A_legs[0][1], A_legs[0][0]) * 57.3)
    rotation_2 = np.abs(np.math.atan2(A_legs[1][0], A_legs[1][1]) * 57.3)
    rot_max = max(rotation_2, rotation_1)
    return rot_max <= rotation_tresh

def is_match_rotation_torso(model, input,rotation_tresh):
    primary_torso = model[2:8]
    secondary_torso = input[2:8]
    (modelTransform_torso, A_torso) = calcTransformationMatrix.calcTransformationMatrix(primary_torso, secondary_torso)
    rotation_1 = np.abs(np.math.atan2(-A_torso[0][1], A_torso[0][0]) * 57.3)
    rotation_2 = np.abs(np.math.atan2(A_torso[1][0], A_torso[1][1]) * 57.3)
    rot_max = max(rotation_2, rotation_1)
    return rot_max <= rotation_tresh

def is_match(model, input):
    #Crop/cut => delen door Xmax & Ymax
    model = normalising.normalise_cte(model)
    input = normalising.normalise_cte(input)

    euclDis_tresh_torso = 0.1
    euclDis_tresh_legs = 0.17
    euclDis_tresh_shoulder = 0.07
    rotation_tresh_legs = 13
    rotation_tresh_torso = 22.56

    #Return final match: alle delen moeten True geven
    return is_match_legs(model,input,euclDis_tresh_legs) and is_match_torso(model,input,euclDis_tresh_torso) and is_match_shoulder (model, input,euclDis_tresh_shoulder) and is_match_rotation_legs(model, input,rotation_tresh_legs) and  is_match_rotation_torso(model, input,rotation_tresh_torso)
