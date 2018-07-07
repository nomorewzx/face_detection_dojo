import numpy as np
import os

LIGHT_GREEN_COLOR_NUMPY_ARRAY = np.array([0, 255, 0], dtype=np.uint8)

# DIR CONFIG

# DEBUG RESULT DIR CONFIG
DEBUG_RESULT_BASE_DIR = os.path.join(os.path.dirname(__file__), './debug_result/')
DEBUG_RESULT_BOXED_FACE_DIR = os.path.join(DEBUG_RESULT_BASE_DIR, 'boxed_faces')
DEBUG_RESULT_LOADED_FACES_DIR = os.path.join(DEBUG_RESULT_BASE_DIR, 'loaded_faces')

FACE_DIR = '~/MyProject/imgs/faces/real_face/'
RESULT_DIR = '~/MyProject/imgs/detect_result/'

DEBUG_FACE_DETECT = True