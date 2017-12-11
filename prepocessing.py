import numpy as np

def split_in_face_legs_torso(features):
    # torso = features[2:8]   #zonder nek
    torso = features[1:8]   #met nek  => if nek incl => compare_incl_schouders aanpassen!!
    legs = features[8:14]
    face = np.vstack([features[0], features[14:18]])

    return (face, torso, legs)

def unsplit(face, torso, legs):
    whole = np.vstack([face[0], torso, legs, face[1:5]])

    return whole

def unpad(matrix):

    return matrix[:, :-1]

def pad(matrix):
    return np.hstack([matrix, np.zeros((matrix.shape[0], 1))])