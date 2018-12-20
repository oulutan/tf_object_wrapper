# tf_object_wrapper
This repository is a wrapper on top of the Tensorflow object detection API. It solves some of its issues and has easy to use scripts to quickly run the detectors.

* Also this repo provides easy ways to use the object detectors and visualize them. Reading and visualizing requires opencv

## Installation ##

Compile and add object_detection API to PYTHONPATH
1. Compile the object detection protos. If you get errors, follow the instructions within the object_detection part of tf models.
``` bash
protoc object_detection/protos/*.proto --python_out=.
``` 
2. Add to python path
``` bash
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
``` 
3. Test installation
``` bash
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
``` 

* Reading, visualizing and writing images require opencv. 
* Tested on Python 2.7 and OpenCV 3.3.0

## Usage ##

