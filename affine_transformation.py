import numpy as np
import logging
logger = logging.getLogger("pose_match")

def find_transformation(model_features, input_features):
    # Zoek 2D affine transformatie matrix om scaling, rotatatie en translatie te beschrijven tussen model en input
    # 2x2 matrix werkt niet voor translaties

    # Pad the data with ones, so that our transformation can do translations too
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])  # horizontaal stacken
    unpad = lambda x: x[:, :-1]


    # It needs to be checked a (0,0) row is present due to undetected body-parts
    # Namely, (0,0) rows a accepted in the input pose
    input_counter = 0

    # List with indices of all the (0,0)-rows
    # This is important because they need to
    # be removed before finding the affine transformation
    # But before returning to caller, they should be restored at the same place.
    # Because the correspondence of the points needs to be preserved
    nan_indices = []

    #print("inputttt: " , input_features)

    input_features_zonder_nan = []
    model_features_zonder_nan = []
    for in_feature in input_features:
        if (in_feature[0] == 0) and (in_feature[1] == 0): #np.isnan(in_feature[0]) and np.isnan(in_feature[1])
            nan_indices.append(input_counter)
        else:
            input_features_zonder_nan.append([in_feature[0], in_feature[1]])
            model_features_zonder_nan.append([model_features[input_counter][0], model_features[input_counter][1]])
        input_counter = input_counter+1

    input_features = np.array(input_features_zonder_nan)
    model_features = np.array(model_features_zonder_nan)

    # padden:
    # naar vorm [ x x 0 1]
    Y = pad(model_features)
    X = pad(input_features)

    # Solve the least squares problem X * A = Y
    # to find our transformation matrix A and then we can display the input on the model = Y'
    A, res, rank, s = np.linalg.lstsq(X, Y)


    transform = lambda x: unpad(np.dot(pad(x), A))
    input_transform = transform(input_features)
    # Restore the (0,0) rows
    # TODO: maybe too much looping ..
    # TODO: convert van matrix->list->matrix ?? crappy
    input_transform_list  = input_transform.tolist()
    for index in nan_indices:
        input_transform_list.insert(index, [0,0])
    input_transform = np.array(input_transform_list)

    #print("traaaans: ", input_transform)
    A[np.abs(A) < 1e-10] = 0  # set really small values to zero

    return (input_transform, A)



#TODO oude functie voor case waar enkel transformatie voor de fixed-points wordt berekend. (OUD)
def calcTransformationMatrix_fixed_points(model, input, secondary):

    # Zoek 2D affine transformatie matrix om scaling, rotatatie en translatie te beschrijven tussen model en input
    # 2x2 matrix werkt niet voor translaties
    # Pad the data with ones, so that our transformation can do translations too
    n = model.shape[0]
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])  # horizontaal stacken
    unpad = lambda x: x[:, :-1]

    # padden:
    # naar vorm [ x x 0 1]
    X = pad(model)
    Y = pad(input)

    # print(X)
    # print(Y)

    # Solve the least squares problem X * A = Y
    # to find our transformation matrix A
    A, res, rank, s = np.linalg.lstsq(X, Y)

    transform = lambda x: unpad(np.dot(pad(x), A))
    #modelTransform = transform(model)
    modelTransform = transform(secondary)

    A[np.abs(A) < 1e-10] = 0  # set really small values to zero

    return (modelTransform, A)