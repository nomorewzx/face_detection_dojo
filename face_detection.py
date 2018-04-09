from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import align.detect_face
from settings import *

def load_and_align_data(image_paths, image_size, margin):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    processed_img_list = []
    tmp_image_paths = image_paths.copy()
    img_list = []
    for image_path in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image_path), mode='RGB')
        processed_img = img.copy()
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
            print("can't detect face")
        for bb in bounding_boxes:
            prewhitened = get_prewhitenned_face_img(bb, image_size, img, img_size, margin, processed_img)
            img_list.append(prewhitened)

        processed_img_list.append(processed_img)

    images = np.stack(img_list)
    processed_imgs = np.stack(processed_img_list)
    return images, processed_imgs


def get_prewhitenned_face_img(bounding_box, image_size, img, img_size, margin, processed_img):
    det = np.squeeze(bounding_box[0:4])
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0] - margin / 2, 0)
    bb[1] = np.maximum(det[1] - margin / 2, 0)
    bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
    bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]

    processed_img[bb[1] - margin:bb[1], bb[0]:bb[2], :] = LIGHT_GREEN_COLOR
    processed_img[bb[3] - margin:bb[3], bb[0]:bb[2], :] = LIGHT_GREEN_COLOR
    processed_img[bb[1]:bb[3], bb[0]-margin: bb[0], :] = LIGHT_GREEN_COLOR
    processed_img[bb[1]:bb[3], bb[2]-margin:bb[2], :] = LIGHT_GREEN_COLOR

    aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
    prewhitened = prewhiten(aligned)
    return prewhitened    

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

if __name__ == '__main__':
    img_dir = '/home/zeks/MyProject/imgs/faces/real_face'
    print('loading...')
