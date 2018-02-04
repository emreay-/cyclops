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
    "number_of_particles": 1000,
    "process_covariance": [1e-2, 1e-2, 1e-2],
    "measurement_covariance": [1e-2, 1e-2, 1e-2],
    "inital_location_covariance": [1e-2, 1e-2, 1e-2],
    "reference_distance": 5.0
 }''')


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
