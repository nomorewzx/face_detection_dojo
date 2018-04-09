from face_detection import load_and_align_data
from utils import load_model_into_tf_session, load_labeled_faces
import tensorflow as tf
import numpy as np
from scipy import misc
from settings import *
import argparse
import sys


def compare(model_path, to_recognize_image_file, image_size, margin):
    names, identity_img_paths = load_labeled_faces()

    face_imgs, _ = load_and_align_data(identity_img_paths, image_size, margin)

    faces_to_identify, _ = load_and_align_data([to_recognize_image_file], image_size, margin)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            load_model_into_tf_session(model_path)

            imgs_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")

            embedding_op = tf.get_default_graph().get_tensor_by_name('embeddings:0')

            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')

            feed_dict = {imgs_placeholder: face_imgs, phase_train_placeholder: False}

            face_embeddings = sess.run(embedding_op, feed_dict = feed_dict)

            face_to_recognize_embeddings = sess.run(embedding_op, feed_dict = {imgs_placeholder: faces_to_identify, phase_train_placeholder: False})


            for i in range(face_to_recognize_embeddings.shape[0]):
                dist = np.sqrt(np.sum(np.square(np.subtract(face_to_recognize_embeddings[i,:], face_embeddings)), axis=1))
                print('Distance Matrix')
                print(' ', end='')
                for name in names:
                    print(' ' + name + ' ', end='')
                print('')
                print(' ', end='')
                for m in range(dist.shape[0]):
                    print(' %1.4f   ' % dist[m], end='')
                print('')
                index = np.argmin(dist)
                print('this is most likely:    ' + names[index])
                # misc.imsave(os.path.join(result_dir, names[index]+'_'+str(i)+'.jpg'), faces_to_identify[i])

def main(args):
    compare(args.model, args.image_file, 160, 5)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Directory contains the model ckpt file')
    parser.add_argument('image_file', type=str, help='Image to recognize')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
    # command with param: python compare.py ~/model_dir ~/MyProject/imgs/faces/real_face/img_to_recognize.jpg
