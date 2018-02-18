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

Cyclops requires a full installation of OpenCV. You can refer the [official documentation](https://docs.opencv.org/3.4.0/d7/d9f/tutorial_linux_install.html) or [here](https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/) for further information. 

Other dependencies can be installed using:
```make installDependencies```

## Deploy

```
cd /path/to/cyclops_top_level
source environment 
cd $CYCLOPS_DIR
python3 user_interface.py
```

