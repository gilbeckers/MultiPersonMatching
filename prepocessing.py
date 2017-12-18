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
