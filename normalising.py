import numpy as np


#Cut pose out of image
def feature_scaling(input):
    xmax = max(input[:, 0])
    ymax = max(input[:, 1])

    xmin = min(input[:, 0])
    ymin = min(input[:, 1])

    sec_x = (input[:, 0]-xmin)/(xmax-xmin)
    sec_y = (input[:, 1]-ymin)/(ymax-ymin)

    #sec_x = (input[:, 0]) / (xmax)
    #sec_y = (input[:, 1]) / (ymax)

    output = np.vstack([sec_x, sec_y]).T

    return output

def feature_scaling_multi_person():

    return


def divide_by_max(input):
    xmax = max(input[:, 0])
    ymax = max(input[:, 1])

    xmin = min(input[:, 0])
    ymin = min(input[:, 1])

    #sec_x = (input[:, 0]-xmin)/(xmax-xmin)
    #sec_y = (input[:, 1]-ymin)/(ymax-ymin)

    sec_x = (input[:, 0]) / (xmax)
    sec_y = (input[:, 1]) / (ymax)

    output = np.vstack([sec_x, sec_y]).T

    return output




def normalise_rescaling(input):
    xmax = max(input[:, 0])
    xmin = min(input[:, 0])
    ymax = max(input[:, 1])
    ymin = min(input[:, 1])

    sec_x = (input[:, 0] - xmin) / (xmax - xmin)
    sec_y = (input[:, 1] - ymin) / (ymax - ymin)
    output = np.vstack([sec_x, sec_y]).T

    return output

def normalise_standardization(input):
    xmean = input[:,0].mean(axis=0)
    ymean = input[:,1].mean(axis=0)
    xstd = np.std(input[:,0])
    ystd = np.std(input[:, 1])

    sec_x = (input[:, 0] - xmean) / xstd
    sec_y = (input[:, 1] - ymean) / ystd
    output = np.vstack([sec_x, sec_y]).T

    return output

