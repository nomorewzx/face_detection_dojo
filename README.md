This repo is inspired by [face_net](https://github.com/davidsandberg/facenet).

Use Python 3.5 or above

This repo contains two features:
1. Face detection and Face alignment
2. Face verification (compare) (Doing)

###Face detection and Face alignment

[MTCNN](https://arxiv.org/pdf/1604.02878.pdf) is used to perform face detection and alignment tasks.
To run this feature on terminal, using commands below:
1. `cd YOUR_PATH/face_detect_dojo/`
2. `python -m verification.face_detection`

Or you can simply open `verification/face_detection.py` file in PyCharm and run this script.

What this script does is to find `test.jpg` in `face_detect_dojo/test_images/` and try to detect the faces (if any) in the image. So please make sure there is an image named `test.jpg` in the dir.

You can see the detection and alignment result in dir `face_detect_dojo/debug_result/boxed_faces/`, in which an image file whose name starts with `BoxedFaces` will exist.

###Face verification (compare) (Doing)

This feature is still on going.

