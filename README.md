# cyclops

[![Build Status](https://travis-ci.org/emreay-/cyclops.svg?branch=master)](https://travis-ci.org/emreay-/cyclops)

Cyclops is a particle filter based planar localization system which only uses a single overhead camera such as a webcam. Currently only a single target (which is called a member) can be localized and tracked, however it is planned to support multiple targets. Here is a simple demo of how it works:


![Alt Text](https://github.com/emreay-/cyclops/blob/master/media/demo.gif)

## System

Cyclops is developed and tested in Ubuntu 16.04 with python 3.6.2 (Anaconda), its portability to other environments has not yet been tested.

## Installing

Just clone this repo:
```git clone https://github.com/emreay-/cyclops.git```

## Dependencies

Cyclops requires a full installation of OpenCV. You can refer to the [official documentation](https://docs.opencv.org/3.4.0/d7/d9f/tutorial_linux_install.html) or [here](https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/) for further information. 

Other dependencies can be installed using:
```make installDependencies```

## Deploy

```
$ cd /path/to/cyclops_top_level
$ source environment 
$ cd $CYCLOPS_DIR
$ python3 user_interface.py
```

## Using

### Camera Calibration

In order to estimate the pose accurately, the used overhead camera should be calibrated (i.e. camera matrix and distortion coefficients should be known). This is fairly a simple process. Please refer: [OpenCV Camera Calibration](https://docs.opencv.org/3.1.0/dc/dbb/tutorial_py_calibration.html), [camera_calibration](https://pypi.python.org/pypi/camera_calibration). 

## Scale Estimation

Cyclops requires an A4 sized paper to estimate the scale (i.e. the number of pixels per meter [px/m]). Simply place the paper somewhere the camera can see, start cyclops with and navigate to `Get Scale` in the user interface. Drag a rectangle around the paper in the popped window. Then press `Space` and then `c` to see the results (i.e. two new popped up windows where they show the cropped rectange and detected paper contour with sizes). Then press `Enter` to continue. If you made a mistake, you can repeat the steps. 