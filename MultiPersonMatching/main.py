from scipy.constants.codata import precision

import pose_match
import parse_openpose_json
import numpy as np
import logging
import prepocessing
import dataset_actions
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
path = '/media/jochen/2FCA69D53AB1BFF49/dataset/poses/'


dataset_actions.statistics()
