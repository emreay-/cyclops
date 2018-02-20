# cyclops

[![Build Status](https://travis-ci.org/emreay-/cyclops.svg?branch=master)](https://travis-ci.org/emreay-/cyclops)

Cyclops is a particle filter based planar localization system which only uses a single overhead camera such as a webcam. Currently only a single target (which is called a member) can be localized and tracked, however it is planned to support multiple targets. Cyclops has a pose server so that the pose estimates can be accessed within the network. 

Here is a simple demo of how it works:


![Alt Text](https://github.com/emreay-/cyclops/blob/master/media/demo.gif)

## System

Cyclops is developed and tested in Ubuntu 16.04 with python 3.6.2 (Anaconda), its portability to other environments has not yet been tested.

## Install

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

## Usage

### Camera Calibration

In order to estimate the pose accurately, the used overhead camera should be calibrated (i.e. camera matrix and distortion coefficients should be known). This is fairly a simple process. Please refer: [OpenCV Camera Calibration](https://docs.opencv.org/3.1.0/dc/dbb/tutorial_py_calibration.html), [camera_calibration](https://pypi.python.org/pypi/camera_calibration). 

The camera calibration file should be in `YAML` format and be placed as `cyclops/param/camera.yaml`. If you want to keep it elsewhere or with a different name, you can modify the corresponding environment variable in `cyclops/environment` file and re-source it.

### Scale Estimation

Cyclops requires an A4 sized paper to estimate the scale (i.e. the number of pixels per meter [px/m]). Simply place the paper somewhere the camera can see, start Cyclops and navigate to `Get Scale` in the user interface. Drag a rectangle around the paper in the popped window. Then press `Space` and then `c` to see the results (i.e. two new popped up windows where they show the cropped rectange and detected paper contour with sizes). Then press `Enter` to continue. If you made a mistake, you can repeat the steps. 

### Members

The targets with unique color pairs are called `Members`. Currently only a single member can be tracked and localized. Each member should have a tag with two distinct and unique colors, one for front and rear. You can simply print two shapes next to each other with colors of choice. The shapes do not matter, the important thing is the distance between the centers of two shapes. You should modify the `reference_distance` parameter under `cyclops/param/filter_parameters.json` file accordingly. Remember that its units are in meters. A sample color tag I used can be found at `cyclops/media/sample_color_tag.pdf`.

### Adding Members

Adding a member means getting the RGB values of the front and rear colors of each member's tag. Due to changes in room light conditions, sensor quality of the used camera and so on, we do not pass the RGB values of the tags with the parameters file. Instead they are taken from the manually selected regions in the image. 

Navigate to the `Add Member` button in the user interface, you will see a window popping up. Drag a rectangle that is small enough to capture only the front color of the member's tag. Press `Space`, `c` and `Enter` respectively. The mean color in the selected area will be taken as the front color. After that, *you should wait for a second window to pop up*. This time, do the same process for the rear color of the tag. Once you finish, you should see a circle in the footer of the user interface with front and rear colors filling the upper and lower halves respectively. 

### Initializing

Optionally, you can select are region for the member's initial location. Navigate to the `Initialize` button in the user interface and drag a rectangle in the popped up window around the member. The center of the circle would be the mean of the intial location distribution. Initial location distribution is a 2D Gaussian and you can adjust the covariance matrix in the `cyclops/param/filter_parameters.json` file. Regardless of manually selecting an initial location or not, the particle orientations are distributed uniformly within `[0, 2*pi)`.

If the initialization part is skipped, then the particle filter would perform a global localization by uniformly distributing the particle locations within the configuration space.

### Start & Reset

Once the scale is known and the members are added, you can start the particle filter. In a successful setup, you will see that the particles are converged on and around the front color of the tag and the arrow which represents the median of the belief is headed with the correct angle. 

It is possible that due to sudden changes in the room light condition or so on, the particle deprivation might occur (i.e. there would be no particles around the true state). In case of particle deprivation, or simply just for testing you can use the `Reset` button to uniformly re-distribute the particles in the configuration space.

### Pose Server

Cyclops has a pose server to make the estimated pose information accessible within the network. This is useful for example, to send the pose info to your robot. When you start the particle filter, the pose server also starts working on a separate thread. The server would have the address of the hostname of your machine and the port number which you can set in the `cyclops/param/filter_parameters.json` file, for example `192.168.0.18:42420`. I'm quite novice in network topics, but if you do not have a DNS in your network to resolve hostnames, you might need to update your hostname's address in `/etc/hosts` file with your local ip which you can find using `ifconfig` (this is what worked for me at least, do it at your own risk). You can take a look in the script `cyclops/cyclops/communication.py` to know how to query the pose server.


### To Do

* Supporting localization and tracking of multiple members concurrently
* Being able to send target locations over network (i.e. to a robot) either from command line or dragging a region in the image
