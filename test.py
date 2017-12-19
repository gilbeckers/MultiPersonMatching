import matplotlib.pyplot as plt
import logging
import matplotlib
print( "version: ", matplotlib.__version__)

logger = logging.getLogger("pose_match")
model_image = plt.imread('data/image_data/foto1.jpg')

