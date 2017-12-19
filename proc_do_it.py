import matplotlib.pyplot as plt
import parse_openpose_json
import prepocessing
import numpy as np

'''
Used in multi_person2():
Superimposes the modelpose on his matching input pose (in the input image space)
This is done with a translation and a rotation an again a translation in the y direction (align feet):

1. Center of gravity are aligned: model pose is superimposed onto input pose:
    Translation & rotation

2. Finally, the feet are aligned 
    Second Translation

'''
def superimpose(input, model, input_image, model_image):

    #input = prepocessing.unpad(input)
    #model = prepocessing.unpad(model)

    # First translation and rotation, NO SCALING
    (d, Z, m) = procrustes(input, model, False) #Scaling is false

    # Zoeken naar laagste punt van lichaam (linker of rechter voet)
    # =>   Max van linker en recht voet (y-coordinaat)
    # => pose wordt aligned met laagste punt (puur translatie)
    voet_index = None

    if input[10][1] >= input[13][1]:
        voet_index = 10
    else:
        voet_index = 13

    # Second translation
    translatie_factor = input[voet_index][1] - Z[voet_index][1]
    Z[:,1] = Z[:,1] + translatie_factor

    '''
    markersize = 3
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
    ax1.imshow(plt.imread(input_image))
    ax1.set_title(input_image + '(input)')
    ax1.plot(*zip(*input), marker='o', color='r', ls='', label='model', ms=markersize)  # ms = markersize

    ax2.set_title(model_image + '(model)')
    ax2.imshow(plt.imread(model_image))
    ax2.plot(*zip(*model), marker='o', color='r', ls='', label='model', ms=markersize)  # ms = markersize

    ax3.set_title('(model sumperimposed on input)')
    ax3.imshow(plt.imread(input_image))
    ax3.plot(*zip(*Z), marker='o', color='r', ls='', label='model', ms=markersize)  # ms = markersize
    plt.show()
    '''

    return Z

def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform
