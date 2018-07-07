import uuid

import tensorflow as tf
from scipy import misc

from align import mtcnn
from settings import *


def load_face_imgs_from_images(abs_image_paths, face_img_size, margin):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    print('load MTCNN model')
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            pnet, rnet, onet = mtcnn.create_mtcnn(sess, None)

    tmp_image_paths = abs_image_paths.copy()
    face_images = []

    for image_path in tmp_image_paths:
        raw_img = misc.imread(image_path, mode='RGB')

        raw_img_copy = raw_img.copy()

        bounding_boxes, _ = mtcnn.detect_face(raw_img, minsize, pnet, rnet, onet, threshold, factor)

        if len(bounding_boxes) < 1:
            print("can't detect face")

        for bounding_box in bounding_boxes:
            prewhitened_face = prewhiten_face_img(bounding_box, face_img_size, raw_img, margin)
            face_images.append(prewhitened_face)

        if DEBUG_FACE_DETECT is True:
            mark_box_around_faces(bounding_boxes=bounding_boxes, margin=margin, raw_image=raw_img_copy)

    face_images = np.stack(face_images)
    return face_images


def mark_box_around_faces(bounding_boxes, margin, raw_image):
    img_size = np.asarray(raw_image.shape)[0:2]

    for bounding_box in bounding_boxes:
        det = np.squeeze(bounding_box[0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])

        raw_image[bb[1] - margin:bb[1], bb[0]:bb[2], :] = LIGHT_GREEN_COLOR_NUMPY_ARRAY
        raw_image[bb[3] - margin:bb[3], bb[0]:bb[2], :] = LIGHT_GREEN_COLOR_NUMPY_ARRAY
        raw_image[bb[1]:bb[3], bb[0] - margin: bb[0], :] = LIGHT_GREEN_COLOR_NUMPY_ARRAY
        raw_image[bb[1]:bb[3], bb[2] - margin:bb[2], :] = LIGHT_GREEN_COLOR_NUMPY_ARRAY

    misc.imsave(os.path.join(DEBUG_RESULT_BOXED_FACE_DIR, 'BoxedFaces_' + str(uuid.uuid1()) + '.jpg'), raw_image)


def prewhiten_face_img(bounding_box, face_image_size, raw_image, margin):
    raw_img_size = np.asarray(raw_image.shape)[0:2]

    det = np.squeeze(bounding_box[0:4])
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0] - margin / 2, 0)
    bb[1] = np.maximum(det[1] - margin / 2, 0)
    bb[2] = np.minimum(det[2] + margin / 2, raw_img_size[1])
    bb[3] = np.minimum(det[3] + margin / 2, raw_img_size[0])

    cropped_face = raw_image[bb[1]:bb[3], bb[0]:bb[2], :]

    resized_face = misc.imresize(cropped_face, (face_image_size, face_image_size), interp='bilinear')

    processed_face = prewhiten(resized_face)

    return processed_face


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


if __name__ == '__main__':
    image_path = os.path.join(os.path.dirname(__file__), '../test_images/test_crowds.jpg')
    img_path_list = [image_path]

    face_images = load_face_imgs_from_images(img_path_list, 160, 10)

    print(face_images.shape)
