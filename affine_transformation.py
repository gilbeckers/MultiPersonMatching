import numpy as np

def find_transformation(model_features, input_features):
    # Zoek 2D affine transformatie matrix om scaling, rotatatie en translatie te beschrijven tussen model en input
    # 2x2 matrix werkt niet voor translaties

    # Pad the data with ones, so that our transformation can do translations too
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])  # horizontaal stacken
    unpad = lambda x: x[:, :-1]

    # padden:
    # naar vorm [ x x 0 1]
    Y = pad(model_features)
    X = pad(input_features)

    # Solve the least squares problem X * A = Y
    # to find our transformation matrix A and then we can display the input on the model = Y'
    A, res, rank, s = np.linalg.lstsq(X, Y)

    transform = lambda x: unpad(np.dot(pad(x), A))
    input_transform = transform(input_features)

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