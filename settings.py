import numpy as np
import os

LIGHT_GREEN_COLOR_NUMPY_ARRAY = np.array([0, 255, 0], dtype=np.uint8)

# DIR CONFIG
RESOURCE_BASE_DIR = os.path.join(os.path.dirname(__file__), 'resources')

# DEBUG RESULT DIR CONFIG
DEBUG_RESULT_BASE_DIR = os.path.join(RESOURCE_BASE_DIR, './debug_result/')
DEBUG_RESULT_BOXED_FACE_DIR = os.path.join(DEBUG_RESULT_BASE_DIR, 'boxed_faces')
DEBUG_RESULT_NO_FACE_DIR = os.path.join(DEBUG_RESULT_BASE_DIR, 'no_face_imgs')

DEBUG_RESULT_LOADED_FACES_DIR = os.path.join(DEBUG_RESULT_BASE_DIR, 'loaded_faces')

# TENSORFLOW EVENT LOG DIR
TENSORFLOW_EVENT_LOG_DIR = os.path.join(os.path.dirname(__file__), 'resources/event_log')

FACE_DIR = '~/MyProject/imgs/faces/real_face/'
RESULT_DIR = '~/MyProject/imgs/detect_result/'

DEBUG_FACE_DETECT = True
DEBUG_TENSORFLOW_GRAPH = True