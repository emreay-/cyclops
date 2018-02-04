import os
import json
import yaml
import math

import numpy as np
import cv2

from cyclops.type_hints import *


class ParticleFilter(object):

    def __init__(self, parameters_file: str, camera_parameters_file: str, 
                 camera_scale: float, color_to_track: color_type):
        self.parameters_file = parameters_file
        self.camera_parameters_file = camera_parameters_file
        self.camera_scale = camera_scale
        self.color_to_track = color_to_track
        
        self.loader(self.check_parameters_file, self.load_parameters)
        self.loader(self.check_camera_parameters_file, self.load_camera_parameters)
    
    @staticmethod
    def loader(condition_to_load: Callable, load_function: Callable, 
               exception: Exception = FileNotFoundError):
        if condition_to_load():
            load_function()
        else:
            raise exception()

    def check_parameters_file(self):
        return os.path.exists(self.parameters_file)

    def load_parameters(self):
        with open(self.parameters_file, 'r') as handle:
            parameters = json.load(handle)
            self.camera_source = parameters['camera_source']
            self.image_size = parameters['image_size']
            self.frame_translation = list(map(lambda x: x//2, self.image_size))
            self.number_of_particles = parameters['number_of_particles']
            self.process_covariance = np.diag(parameters['process_covariance'])
            self.measurement_covariance = np.diag(parameters['measurement_covariance'])
            self.inital_location_covariance = np.diag(parameters['inital_location_covariance'])
            self.reference_distance = parameters['reference_distance']

    def check_camera_parameters_file(self):
        return os.path.exists(self.camera_parameters_file)

    def load_camera_parameters(self):
        with open(self.camera_parameters_file,'r') as handle:
            camera_info = yaml.load(handle)
            camera_matrix = camera_info['camera_matrix']['data']
            self.camera_matrix = np.reshape(np.array(camera_matrix),(3,3))
            distortion = camera_info['distortion_coefficients']['data']
            self.distortion = np.array(distortion)

    def initialize_particles(self, start_location: coordinate):
        start_location = self.convert_pixel_space_to_physical_space(start_location)
        self.particles = np.transpose(np.random.multivariate_normal(
            list(start_location) + [0.0], self.inital_location_covariance, self.number_of_particles))
        self.apply_mode_to_particle_thetas()
        self.weights = np.array([1.0] * self.number_of_particles)

    def convert_pixel_space_to_physical_space(self, location: coordinate):
        x = location[0] * self.camera_scale + self.frame_translation[0]
        y = -1 * location[1] * self.camera_scale + self.frame_translation[1]
        return x, y

    def apply_mode_to_particle_thetas(self):
        self.particles[2, :] = np.apply_along_axis(
            lambda x: x % (math.pi * 2), axis=0, arr=self.particles[2, :])

    def run(self):
        self.create_video_capture()
        while True:
            self.get_undistorted_image()
            self.run_process_model()
            self.run_measurement_update()
            self.resample_particles()
            self.wait_key()
    
    def create_video_capture(self):
        self.video_capture = cv2.VideoCapture(self.camera_source)

    def get_undistorted_image(self):
        self.capture_image()
        self.undistort_image()

    def capture_image(self):
        _ , self.captured_image = self.video_capture.read()

    def undistort_image(self):
        self.undistored_image = cv2.undistort(self.captured_image, self.camera_matrix, self.distortion)

    def run_process_model(self):
        self.process_noise = np.transpose(np.random.multivariate_normal(
            mean=(0., 0., 0.), cov=self.process_covariance, size=self.number_of_particles))
        self.particles += self.process_noise
    
    def run_measurement_update(self):
        pass

    def convert_particles_to_pixel_space(self):
        self.particle_pixel_coordinates = self.particles
        self.particle_pixel_coordinates[0, :] = np.apply_along_axis(
            lambda x: np.floor((x - self.frame_translation[0]) / (-1 * self.camera_scale)), axis=0, arr=self.particles[1, :])
        print(self.particle_pixel_coordinates)

    def resample_particles(self):
        pass

    def wait_key(self):
        return cv2.waitKey(50)