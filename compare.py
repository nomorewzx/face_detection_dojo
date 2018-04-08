from face_detection import load_and_align_data
from utils import load_model_into_tf_session, load_labeled_faces
import tensorflow as tf
import numpy as np
from scipy import misc
from settings import *
import os

def compare(model_path, image_files, image_size, margin):
    names, face_imgs = load_labeled_faces()
    result_dir  = os.path.expanduser(RESULT_DIR)
    faces_to_identify, processed_image = load_and_align_data(image_files, image_size, margin)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            load_model_into_tf_session(model_path)

            imgs_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")

            embedding_op = tf.get_default_graph().get_tensor_by_name('embeddings:0')

            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')

            feed_dict = {imgs_placeholder: face_imgs, phase_train_placeholder: False}

            face_embeddings = sess.run(embedding_op, feed_dict = feed_dict)

            face_to_identify_embeddings = sess.run(embedding_op, feed_dict = {imgs_placeholder: faces_to_identify, phase_train_placeholder: False})


            for i in range(face_to_identify_embeddings.shape[0]):
                dist = np.sqrt(np.sum(np.square(np.subtract(face_to_identify_embeddings[i,:], face_embeddings)), axis=1))
                index = np.argmax(dist)
                print(names[index])
                misc.imsave(os.path.join(result_dir, names[index]+'_'+str(i)+'.jpg'), faces_to_identify[i])

def main():
    pass

if __name__ == '__main__':
    model_path = '~/MyProject/pre_trained_models/20170511-185253/'

    image_files = ['~/MyProject/imgs/to_detect/zhaoshaoyi_1.jpg']

    image_size = 160

    margin = 5

    gpu_memory_fraction = 1.0

    compare(model_path, image_files, image_size, margin)
