# cv_dnn_face
Sample scripts using OpenCV DNN based face detection and recogniton.

## Environment
- Ubuntu 20.04.4 LTS
- Python 3.8.10

## requirements
- opencv-python >= 4.5.4
- numpy

## model files
Download following onnx model files and locate in model direcory.
model/yunet.onnx
model/face_recognizer_fast.onnx

## demo scripts
### detector.py

You can show face detection demo.

```
$ python detector.py -h
usage: detector.py [-h] img_file

face_recognizer

positional arguments:
  img_file    image_file or camera_number

optional arguments:
  -h, --help  show this help message and exit
```

### recognizer.py

You can show face recognition demo.
This assumes you have face features in aligned_faces directory.


```
$ python recognizer.py -h
usage: recognizer.py [-h] img_file

face_recognizer

positional arguments:
  img_file    image_file or camera_number

optional arguments:
  -h, --help  show this help message and exit

```

### How to generate face samples and face dataset in aligned_faces directory.
### crop.py

This will crop faces with margin.

```commandline
$ python crop.py -h
usage: crop.py [-h] [--clockwise] dir

face cropper from images.

positional arguments:
  dir          image source dir

optional arguments:
  -h, --help   show this help message and exit
  --clockwise  rotate image clockwize

```

### crop_from_video.py
This will crop faces with margin from video.

```commandline
$ python crop_from_video.py  -h
usage: crop_from_video.py [-h] [--dst_dir DST_DIR] [--interval INTERVAL] name

crop face from video

positional arguments:
  name                 video file

optional arguments:
  -h, --help           show this help message and exit
  --dst_dir DST_DIR    dst dir
  --interval INTERVAL  frame interval to process

```


### move_similar_faces.py
You might want to remove too similar faces to reduce dataset.

```commandline

$ python move_similar_faces.py -h
usage: move_similar_faces.py [-h] [--th TH] [-r] src_dir dst_dir

face cropper from images.

positional arguments:
  src_dir     image source dir
  dst_dir     destination dir

optional arguments:
  -h, --help  show this help message and exit
  --th TH     if face_distance is smaller than threshold, skips
  -r          recursive file search
```

### generate_aligned_faces.py

This will generate feature data file (*.npy), and face image without margin (*.jpg)

```commandline

$ python generate_aligned_faces.py -h
usage: generate aligned face images from an image [-h] image

positional arguments:
  image       input image file path (./image.jpg)

optional arguments:
  -h, --help  show this help message and exit
```

Note:
recognizer.py assumes feature file name with label.
biden.npy means label is biden.

## Download
--------
[yunet.onnx](https://github.com/ShiqiYu/libfacedetection.train/blob/master/tasks/task1/onnx/yunet.onnx)
[face_recognizer_fast.onnx](https://drive.google.com/file/d/1ClK9WiB492c5OZFKveF3XiHCejoOxINW/view?usp=sharing)

## SeeAlso
https://github.com/opencv/opencv/blob/4.x/samples/dnn/README.md
https://github.com/opencv/opencv/blob/4.x/samples/dnn/face_detect.py

This was inspired by following URL
https://qiita.com/UnaNancyOwen/items/f3db189760037ec680f3
