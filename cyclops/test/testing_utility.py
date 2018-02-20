import os
import sys
import unittest
import tempfile
import shutil

from subprocess import check_call

import numpy as np
import cv2

from cyclops.type_hints import *


def remove_files_if_exist(*args):
    for file in args:
        if exists(file):
            remove_file(file)


def exists(file_path: str):
    return os.path.exists(file_path)


def remove_file(file_path: str):
    os.remove(file_path)


def remove_directories_if_exist(*args):
    for directory in args:
        if exists(directory):
            remove_directory(directory)


def remove_directory(directory: str):
    shutil.rmtree(directory)


class CyclopsUnitTestBase(unittest.TestCase):

    pass        


def write_dummy_parameters_file(*args):
    for arg in args:
        with open(arg, 'w') as handle:
            handle.writelines('''{
    "camera_source": 0,
    "image_width": 640,
    "image_height": 480,
    "number_of_particles": 1000,
    "process_covariance": [1e-2, 1e-2, 1e-2],
    "measurement_covariance": [1e-2, 1e-2, 1e-2],
    "inital_xy_covariance": [1e-2, 1e-2],
    "reference_distance": 0.05,
    "server_port": 42420
 }''')


def write_dummy_camera_parameters_file(*args):
    for arg in args:
        with open(arg, 'w') as handle:
            handle.writelines('''image_width: 640
image_height: 480
camera_name: narrow_stereo
camera_matrix:
  rows: 3
  cols: 3
  data: [696.019484, 0.000000, 349.918582, 0.000000, 696.225875, 217.635331, 0.000000, 0.000000, 1.000000]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [0.259910, -0.975120, -0.021119, -0.007444, 0.000000]
rectification_matrix:
  rows: 3
  cols: 3
  data: [1.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 1.000000]
projection_matrix:
  rows: 3
  cols: 4
  data: [705.152710, 0.000000, 348.171845, 0.000000, 0.000000, 705.937256, 211.281035, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000]''')


def create_dummy_video_file(output_dir: str, file_name_without_extension: str = 'out', 
                            fps: float = 20.0, video_size: pixel_coord = (480, 640),
                            video_length_in_seconds: float = 2.0):

    _video_path = os.path.join(output_dir, file_name_without_extension + '.mp4')
    _number_of_frames = round(video_length_in_seconds * fps)
    _frame = np.zeros((video_size[0], video_size[1], 3), np.uint8)

    for i in range(_number_of_frames):
        _png_path = os.path.join(output_dir, 'img{:05d}.png'.format(i))
        cv2.imwrite(_png_path, _frame)        
    
    _image = output_dir + os.path.sep + 'img%05d.png'

    command = ['ffmpeg', '-loglevel', 'quiet', '-y', '-f', 'image2', '-r', '24',
               '-i', _image, '-an', '-vcodec', 'mpeg4', _video_path ]
    check_call(command)

    return _video_path
