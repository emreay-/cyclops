import os
import json
import yaml
import math
import logging

import numpy as np
import cv2
from scipy.stats import multivariate_normal

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
            self.image_width = parameters['image_width']
            self.image_height = parameters['image_height']
            self.frame_translation = [self.image_width // 2, self.image_height // 2]
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
        logging.debug('Particle Initialization')
        start_location = self.convert_pixel_space_to_physical_space(start_location)
        self.particles = np.transpose(np.random.multivariate_normal(
            list(start_location) + [0.0], self.inital_location_covariance, self.number_of_particles))
        self.apply_mode_to_particle_thetas()
        self.weights = np.array([1.0] * self.number_of_particles)

    def convert_pixel_space_to_physical_space(self, location: coordinate):
        x = (location[0] - self.frame_translation[0]) / self.camera_scale
        y = (-1) * (location[1] + self.frame_translation[1]) / self.camera_scale
        return x, y

    def apply_mode_to_particle_thetas(self):
        self.particles[2, :] = np.apply_along_axis(
            lambda x: x % (math.pi * 2), axis=0, arr=self.particles[2, :])

    def run(self):
        self.create_video_capture()
        self.create_window()
        self.create_density_functions()
        while True:
            self.get_undistorted_image()
            self.convert_particles()
            self.run_process_model()
            self.run_measurement_update()
            self.resample_particles()
            self.visualize()
            key = self.wait_key()
    
    def create_video_capture(self):
        self.video_capture = cv2.VideoCapture(self.camera_source)

    def create_window(self):
        self.window = 'PF Localization'
        cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)

    def create_density_functions(self):
        self.measurement_probability_for_xy = multivariate_normal(
            mean=self.color_to_track, cov=self.measurement_covariance)
        
        self.measurement_probability_for_theta = multivariate_normal(
            mean=(0, 0, 0), cov=self.measurement_covariance)

    def get_undistorted_image(self):
        self.capture_image()
        self.undistort_image()

    def capture_image(self):
        _ , self.captured_image = self.video_capture.read()

    def undistort_image(self):
        self.undistorted_image = cv2.undistort(self.captured_image, self.camera_matrix, self.distortion)

    def convert_particles(self):
        self.particles_in_pixel_space = self.convert_array_from_physical_to_pixel_space(self.particles)

    def run_process_model(self):
        logging.debug('Process')
        self.process_noise = np.transpose(np.random.multivariate_normal(
            mean=(0., 0., 0.), cov=self.process_covariance, size=self.number_of_particles))
        self.particles += self.process_noise
    
    def run_measurement_update(self):
        logging.debug('Measurement Update')
        self.convert_particles()
        measurements_for_xy = self.get_measurements(self.particles_in_pixel_space)
        weights_xy = self.get_probabilities_for_xy_measurements(measurements_for_xy)

        locations_for_angle_measurements = self.get_locations_for_angle_measurements()
        pixel_locations_for_angle_measurement = self.convert_array_from_physical_to_pixel_space(
            locations_for_angle_measurements)
        measurements_for_angle = self.get_measurements(pixel_locations_for_angle_measurement)
        weights_theta = self.get_probabilities_for_angle_measurements(measurements_for_angle)

        self.weights = np.mean(np.array([weights_xy, 0.2*weights_theta]), axis=0)
        # self.weights = weights_xy
        self.normalize_weights()

    def convert_array_from_physical_to_pixel_space(self, array: np.array):
        converted_array = np.copy(array)
        
        converted_array[0, :] = np.apply_along_axis(
            lambda x: x * self.camera_scale + self.frame_translation[0], 
            axis=0, arr=array[0, :])

        converted_array[1, :] = np.apply_along_axis(
            lambda y: -1 * (y * self.camera_scale + self.frame_translation[1]),
            axis=0, arr=array[1, :])
        return converted_array.astype(int)

    def get_image_value_at(self, coord: coordinate):
        (x, y) = coord
        if self.check_pixel_coordinates(x, y):
            return self.undistorted_image[int(y), int(x)]
        else:
            return (-1, -1, -1)
    
    def check_pixel_coordinates(self, x, y):
        return (0 <= x < self.image_width) and (0 <= y < self.image_height)

    def get_measurements(self, array_of_pixel_coordinates: np.array):
        return np.apply_along_axis(self.get_image_value_at, axis=0, arr=array_of_pixel_coordinates[:2, :])

    def get_probabilities_for_xy_measurements(self, measurements: np.array):
        def _probability(measurement):
            if (measurement[0], measurement[1], measurement[2]) == (-1, -1, -1):
                return 0.
            else: 
                return self.measurement_probability_for_xy.pdf((measurement[0], measurement[1], measurement[2])) 
        return np.apply_along_axis(_probability, axis=0, arr=measurements)

    def get_locations_for_angle_measurements(self):
        def _get_location(particle):
            if (particle[2] % math.pi) == 0:
                delta_x = self.reference_distance
                delta_y = 0
            elif (particle[2] % math.pi) == math.pi / 2.:
                delta_x = 0
                delta_y = self.reference_distance
            else:
                delta_x = self.reference_distance * math.cos(particle[2])
                delta_y = self.reference_distance * math.sin(particle[2])
            
            return np.array([particle[0] + delta_x, particle[1] + delta_y])
        
        return np.apply_along_axis(_get_location, axis=0, arr=self.particles)

    def get_probabilities_for_angle_measurements(self, measurements: np.array):
        return np.apply_along_axis(
            lambda m: self.measurement_probability_for_theta.pdf((m[0], m[1], m[2])), axis=0, arr=measurements)

    def normalize_weights(self):
        normalization_factor = np.sum(self.weights)
        self.weights = self.weights / normalization_factor

    def resample_particles(self):        
        logging.debug('Resample')
        weights_cdf = np.cumsum(self.weights)
        random_number = np.random.uniform(0., 1/self.number_of_particles)
        resampled_particle_indeces = []
        for i in range(self.number_of_particles):
            idx = np.min(np.argwhere(weights_cdf >= random_number + (i - 1) / self.number_of_particles))
            resampled_particle_indeces.append(idx)
        self.particles = self.particles[:,resampled_particle_indeces]
        self.apply_mode_to_particle_thetas()

    def visualize(self):
        _image = self.undistorted_image.copy()
        for i in range(self.number_of_particles):
            x, y = self.particles_in_pixel_space[0, i], self.particles_in_pixel_space[1, i]
            if self.check_pixel_coordinates(x, y):
                cv2.circle(_image, center=(x, y), radius=2, color=(100, 250, 0), thickness=-1)
        mean_x = int(np.mean(self.particles_in_pixel_space[0, :]))
        mean_y = int(np.mean(self.particles_in_pixel_space[1, :]))
        mean_theta = -np.mean(self.particles[2, :])
        d = 40
        cv2.line(_image, pt1=(mean_x, mean_y), pt2=(int(mean_x + d*math.cos(mean_theta)), 
            int(mean_y + d*math.sin(mean_theta))), color=(200, 0, 0), thickness=2)

        cv2.imshow(self.window, _image)

    def wait_key(self):
        return cv2.waitKey(50)