import os
import json
import yaml
import math
import logging
import socket
import time

import numpy as np
import cv2
from scipy.stats import multivariate_normal

from cyclops.type_hints import color_type, coordinate, pixel_coord, Callable
from cyclops.communication import PoseServer


class ParticleFilter(object):

    def __init__(self, parameters_file: str, camera_parameters_file: str, 
                 camera_scale: float, front_color: color_type, 
                 rear_color: color_type):
        self.parameters_file = parameters_file
        self.camera_parameters_file = camera_parameters_file
        self.camera_scale = camera_scale
        self.front_color = front_color
        self.rear_color = rear_color
        self.is_in_progress = False

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
            self.number_of_particles = parameters['number_of_particles']
            self.process_covariance = np.diag(parameters['process_covariance'])
            self.measurement_covariance = np.diag(parameters['measurement_covariance'])
            self.inital_xy_covariance = np.diag(parameters['inital_xy_covariance'])
            self.reference_distance = parameters['reference_distance']
            self.server_port = parameters['server_port']

            self.frame_translation = [self.image_width // 2, self.image_height // 2]
            self.configuration_space_x_min, self.configuration_space_y_min = \
                self.convert_pixel_space_to_physical_space((0, 0))
            self.configuration_space_x_max, self.configuration_space_y_max = \
                self.convert_pixel_space_to_physical_space((self.image_width, self.image_height))

    def check_camera_parameters_file(self):
        return os.path.exists(self.camera_parameters_file)

    def load_camera_parameters(self):
        with open(self.camera_parameters_file,'r') as handle:
            camera_info = yaml.load(handle)
            camera_matrix = camera_info['camera_matrix']['data']
            self.camera_matrix = np.reshape(np.array(camera_matrix),(3,3))
            distortion = camera_info['distortion_coefficients']['data']
            self.distortion = np.array(distortion)

    def convert_pixel_space_to_physical_space(self, location: coordinate):
        x = (location[0] - self.frame_translation[0]) / self.camera_scale
        y = (-1 * location[1] + self.frame_translation[1]) / self.camera_scale
        return x, y

    def initialize_particles(self, start_location: pixel_coord = None):
        logging.debug('Particle Initialization')
        self.reset_filter()
        self.distribute_particle_angles_uniformly()

        if start_location:
            self.distribute_particle_positions_with_gaussian(start_location)
        else:
            self.distribute_particle_positions_uniformly()

    def reset_filter(self):
        self.particles = np.zeros((3, self.number_of_particles))
        self.weights = np.ones(self.number_of_particles)

    def distribute_particle_positions_uniformly(self):
        self.particles[0, :] = np.random.uniform(self.configuration_space_x_min, 
                                                 self.configuration_space_x_max, 
                                                 self.number_of_particles)
        self.particles[1, :] = np.random.uniform(self.configuration_space_y_min, 
                                                 self.configuration_space_y_max, 
                                                 self.number_of_particles)

    def distribute_particle_positions_with_gaussian(self, start_location: pixel_coord):
        start_location = self.convert_pixel_space_to_physical_space(start_location)
        self.particles[:2, :] = np.transpose(np.random.multivariate_normal(
            list(start_location), self.inital_xy_covariance, self.number_of_particles))

    def distribute_particle_angles_uniformly(self):
        self.particles[2, :] = np.random.uniform(0.0, math.pi * 2, self.number_of_particles)

    def apply_mode_to_particle_thetas(self):
        self.particles[2, :] = np.apply_along_axis(
            lambda x: x % (math.pi * 2), axis=0, arr=self.particles[2, :])

    def run(self):
        self.pose_server = PoseServer(port=self.server_port)
        self.create_video_capture()
        self.create_window()
        self.create_density_functions()
        self.is_in_progress = True

        while self.is_in_progress:
            self.get_undistorted_image()
            self.convert_particles()
            self.run_process_model()
            self.run_measurement_update()
            self.resample_particles()
            self.compute_belief_from_particles()
            self.visualize()
            self.update_pose_server()
            self.wait_key()
    
    def create_video_capture(self):
        self.video_capture = cv2.VideoCapture(self.camera_source)

    def create_window(self):
        self.window = 'PF Localization'
        cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)

    def create_density_functions(self):
        self.measurement_probability_for_xy = multivariate_normal(
            mean=self.front_color, cov=self.measurement_covariance)
        
        self.measurement_probability_for_theta = multivariate_normal(
            mean=self.rear_color, cov=self.measurement_covariance)

    def create_socket(self):
        self.socket = socket.socket()
        self.server_address = (socket.gethostname(), 12349)
        self.socket.connect(self.server_address)

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

        # self.weights = np.mean(np.array([weights_xy, weights_theta]), axis=0)
        self.weights = np.multiply(weights_xy, weights_theta)
        self.normalize_weights()

    def convert_array_from_physical_to_pixel_space(self, array: np.array):
        converted_array = np.copy(array)
        
        converted_array[0, :] = np.apply_along_axis(
            lambda x: x * self.camera_scale + self.frame_translation[0], 
            axis=0, arr=array[0, :])

        converted_array[1, :] = np.apply_along_axis(
            lambda y: (-1 * y * self.camera_scale) + self.frame_translation[1],
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
            
            return np.array([particle[0] - delta_x, particle[1] - delta_y])
        
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

    def compute_belief_from_particles(self):
        self.median_x = int(np.median(self.particles_in_pixel_space[0, :]))
        self.median_y = int(np.median(self.particles_in_pixel_space[1, :]))
        self.median_theta = -np.median(self.particles[2, :])
        self.median_x_physical, self.median_y_physical = \
            self.convert_pixel_space_to_physical_space((self.median_x, self.median_y))
        self.median_theta_physical = -self.median_theta

    def visualize(self):
        _image = self.undistorted_image.copy()
        _belief_line_length_pixels = 40
        _particle_line_length_pixels = 20

        for i in range(self.number_of_particles):
            x, y, theta = self.particles_in_pixel_space[0, i], \
                          self.particles_in_pixel_space[1, i], \
                          -self.particles_in_pixel_space[2, i]
            if self.check_pixel_coordinates(x, y):
                cv2.arrowedLine(_image, pt1=(x, y), 
                    pt2=(int(x + _particle_line_length_pixels * math.cos(theta)),
                         int(y + _particle_line_length_pixels * math.sin(theta))), 
                    color=(100, 250, 0), thickness=1)
        
        cv2.arrowedLine(_image, 
            pt1=(self.median_x, self.median_y), 
            pt2=(int(self.median_x + _belief_line_length_pixels * math.cos(self.median_theta)), 
                 int(self.median_y + _belief_line_length_pixels * math.sin(self.median_theta))), 
            color=(200, 0, 0), thickness=2)
        
        cv2.putText(_image, '{:.2f}, {:.2f}, {:.1f}'.format(
            self.median_x_physical, self.median_y_physical, 
            math.degrees(self.median_theta_physical)), (self.median_x, self.median_y), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness=2, color=(200, 25, 200))

        cv2.imshow(self.window, _image)

    def update_pose_server(self):
        self.socket = socket.socket()
        self.server_address = (socket.gethostname(), self.server_port)
        
        try:
            self.socket.connect(self.server_address)
            _current_belief_in_bytes = str.encode(str({'purpose': 'update',
                                                    'id': '0', 
                                                    'timestamp': time.time(),
                                                    'x': self.median_x_physical, 
                                                    'y': self.median_y_physical,
                                                    'theta': self.median_theta_physical}))
            self.socket.send(_current_belief_in_bytes)
        except Exception as ex:
            print(ex)
        
        self.socket.close()

    def wait_key(self):
        return cv2.waitKey(50)

    @property
    def is_in_progress(self):
        return self._is_in_progress

    @is_in_progress.setter
    def is_in_progress(self, value: bool):
        self._is_in_progress = value
