import os
import sys
import unittest
import tempfile
import shutil

import numpy as np
import cv2

from cyclops.test.testing_utility import *
from cyclops.particle_filter import *


class ParticleFilterTests(CyclopsUnitTestBase):

    @classmethod
    def setUpClass(cls):
        cls.__parameters_file_object = tempfile.NamedTemporaryFile()
        cls.__parameters_file_path = cls.__parameters_file_object.name
        write_dummy_parameters_file(cls.__parameters_file_path)

        cls.__camera_parameters_object = tempfile.NamedTemporaryFile()
        cls.__camera_parameters_path = cls.__camera_parameters_object.name
        write_dummy_camera_parameters_file(cls.__camera_parameters_path)

        cls.__temp_video_dir = tempfile.mkdtemp()
        cls.__dummy_video_file = create_dummy_video_file(cls.__temp_video_dir)

    def setUp(self):
        self.__camera_scale = 540.0
        self.__color_to_track = (0, 255, 0)
        self.__object = ParticleFilter(parameters_file = self.__parameters_file_path,
                                       camera_parameters_file = self.__camera_parameters_path,
                                       camera_scale = self.__camera_scale,
                                       color_to_track = self.__color_to_track)

        self.__expected_camera_source = 0
        self.__expected_image_width = 640
        self.__expected_image_height = 480
        self.__expected_frame_translation = [320, 240]
        self.__expeced_number_of_particles = 1000
        self.__expected_process_covariance = np.diag([1e-2, 1e-2, 1e-2])
        self.__expected_measurement_covariance = np.diag([1e-2, 1e-2, 1e-2])
        self.__expected_inital_location_covariance = np.diag([1e-2, 1e-2, 1e-2])
        self.__expected_reference_distance = 0.05
        self.__expected_camera_matrix = np.reshape(np.array([696.019484, 0.000000, 349.918582, 
                                                             0.000000, 696.225875, 217.635331, 
                                                             0.000000, 0.000000, 1.000000]), (3, 3))
        self.__expected_distortion = np.array([0.259910, -0.975120, -0.021119, -0.007444, 0.000000])
        self.__start_location = (100., 100.)
        self.__expected_weights = np.ones((self.__expeced_number_of_particles))
        self.__expected_particle_thetas = np.array([ 5.0, 5.5, 6.0, 0.21681469, 0.71681469, 1.21681469])
        self.__particles_before_process = np.array([5.0, 5.5, 6.0, 6.5, 7.0, 7.5])
        
        self.__expected_captured_image = np.zeros((480, 640, 3), np.uint8)
        self.__coordinate_to_convert = (320, 240)
        self.__expected_converted_coordinate = (0., 0.)
        self.__expected_particles_after_conversion = np.array(object=[[ 3020,   3290,   3560,   3830,   4100,   4370],
                                                                      [-2940,  -3210,  -3480,  -3750,  -4020,  -4290],
                                                                      [    5,      5,      6,      6,      7,      7]])
        self.__array_of_pixel_coordinates_to_get_measurements = np.array(object=[[100, 200, 100, 300],
                                                                                 [100, 200, 100, 300],
                                                                                 [5.0, 5.5, 6.0, 6.5]])
        self.__expected_measurements = np.array(object=[[100., 100., 100., 100.],
                                                        [150., 150., 150., 150.],
                                                        [200., 200., 200., 200.]])
        self.__measurements_to_compute_xy_probability = np.repeat(np.array(self.__color_to_track).reshape(3,1), 5).reshape(3, 5)
        self.__expected_probabilities_for_xy_measurements = np.array([63.49363593] * 5)
        self.__expected_locations_for_angle_measurements_theta_0 = np.repeat(
            np.array([[self.__object.reference_distance],[0.]]), 3).reshape(2, 3)
        self.__expected_locations_for_angle_measurements_theta_90 = np.repeat(
            np.array([[0.],[self.__object.reference_distance]]), 3).reshape(2, 3)
        self.__expected_locations_for_angle_measurements_theta_45 = np.repeat(
            np.array([[self.__object.reference_distance * math.cos(math.pi / 4.0)],
                      [self.__object.reference_distance * math.sin(math.pi / 4.0)]]), 3).reshape(2, 3)
        self.__measurements_to_compute_angle_probability = np.zeros((3, 5))
        self.__expected_probabilities_for_angle_measurements = np.array([63.49363593] * 5)
        self.__expected_weights_after_measurement_update = np.array([31.74681797] * 5)

    @classmethod
    def tearDownClass(cls):
        remove_directories_if_exist(cls.__temp_video_dir)

    def reset_particles(self, particle_values: np.array):
        self.__object.particles = np.zeros((3, len(particle_values)), dtype=float)
        self.__object.particles[:] = particle_values
        self.__object.number_of_particles = len(particle_values)

    def mock_video_capture(self):
        self.__object.camera_source = self.__dummy_video_file
        self.__object.create_video_capture()
        self.__object.capture_image()

    def mock_get_undistorted_image(self):
        self.__object.camera_source = self.__dummy_video_file
        self.__object.create_video_capture()
        self.__object.get_undistorted_image()

    def test_init_with_empty_parameters_file(self):
        self.assertRaises(FileNotFoundError, ParticleFilter, tempfile.NamedTemporaryFile().name, 
                          self.__camera_parameters_path, self.__camera_scale, self.__color_to_track)

    def test_init_with_empty_camera_parameters_file(self): 
        self.assertRaises(FileNotFoundError, ParticleFilter, self.__parameters_file_path,
                          tempfile.NamedTemporaryFile().name, self.__camera_scale, self.__color_to_track)

    def test_load_parameters(self):
        self.assertEqual(self.__object.camera_source, self.__expected_camera_source)
        self.assertEqual(self.__object.image_width, self.__expected_image_width)
        self.assertEqual(self.__object.image_height, self.__expected_image_height)
        self.assertListEqual(self.__object.frame_translation, self.__expected_frame_translation)
        self.assertEqual(self.__object.number_of_particles, self.__expeced_number_of_particles)
        np.testing.assert_array_almost_equal(self.__object.process_covariance, self.__expected_process_covariance)
        np.testing.assert_array_almost_equal(self.__object.measurement_covariance, self.__expected_measurement_covariance)
        np.testing.assert_array_almost_equal(self.__object.inital_location_covariance, self.__expected_inital_location_covariance)
        self.assertAlmostEqual(self.__object.reference_distance, self.__expected_reference_distance)

    def test_load_camera_parameters(self):
        np.testing.assert_almost_equal(self.__object.camera_matrix, self.__expected_camera_matrix)
        np.testing.assert_almost_equal(self.__object.distortion, self.__expected_distortion)

    def test_initialize_particles(self):
        self.__object.initialize_particles(self.__start_location)
        self.assertEqual(self.__object.particles.shape[0], 3)
        self.assertEqual(self.__object.particles.shape[1], self.__expeced_number_of_particles)
        np.testing.assert_almost_equal(self.__object.weights, self.__expected_weights)

    def test_apply_mode_to_particle_thetas(self):
        self.reset_particles(self.__particles_before_process)
        self.__object.apply_mode_to_particle_thetas()
        np.testing.assert_array_almost_equal(self.__object.particles[2, :], self.__expected_particle_thetas)
    
    def test_capture_image(self):
        self.mock_video_capture()
        np.testing.assert_almost_equal(self.__object.captured_image, self.__expected_captured_image)

    def test_convert_pixel_space_to_physical_space(self):
        _converted_coordinate = self.__object.convert_pixel_space_to_physical_space(self.__coordinate_to_convert)
        self.assertTupleEqual(_converted_coordinate, self.__expected_converted_coordinate)

    def test_convert_array_from_physical_to_pixel_space(self):
        self.reset_particles(self.__particles_before_process)
        _converted_array = self.__object.convert_array_from_physical_to_pixel_space(self.__object.particles)
        np.testing.assert_almost_equal(_converted_array, self.__expected_particles_after_conversion)

    def test_get_image_value_at(self):
        self.mock_get_undistorted_image()
        _image_value = self.__object.get_image_value_at((100, 100))
        np.testing.assert_almost_equal(_image_value, np.array([0., 0., 0.]))

    def test_get_measurements(self):
        self.__object.undistorted_image = np.full((640, 480, 3), [100, 150, 200], dtype=np.uint8)
        _measurements = self.__object.get_measurements(self.__array_of_pixel_coordinates_to_get_measurements)
        np.testing.assert_almost_equal(_measurements, self.__expected_measurements)

    def test_run_process_model(self):
        self.reset_particles(self.__particles_before_process)
        self.__object.run_process_model()
        self.assertRaises(AssertionError, np.testing.assert_almost_equal, 
                          self.__object.particles, self.__particles_before_process)

    def test_get_probabilities_for_xy_measurements(self):
        self.__object.create_density_functions()
        _probabilities_for_xy_measurements = self.__object.get_probabilities_for_xy_measurements(
            self.__measurements_to_compute_xy_probability)
        np.testing.assert_almost_equal(_probabilities_for_xy_measurements, 
                                       self.__expected_probabilities_for_xy_measurements)

    def test_get_locations_for_angle_measurements_theta_0(self):
        self.reset_particles(np.zeros((3,)))
        _locations_for_angle_measurements = self.__object.get_locations_for_angle_measurements()
        np.testing.assert_almost_equal(
            _locations_for_angle_measurements, self.__expected_locations_for_angle_measurements_theta_0)

    def test_get_locations_for_angle_measurements_theta_90(self):
        self.reset_particles(np.zeros((3,)))
        self.__object.particles[2, :] = np.array([math.pi / 2.0] * 3)
        _locations_for_angle_measurements = self.__object.get_locations_for_angle_measurements()
        np.testing.assert_almost_equal(
            _locations_for_angle_measurements, self.__expected_locations_for_angle_measurements_theta_90)

    def test_get_locations_for_angle_measurements_theta_45(self):
        self.reset_particles(np.zeros((3,)))
        self.__object.particles[2, :] = np.array([math.pi / 4.0] * 3)
        _locations_for_angle_measurements = self.__object.get_locations_for_angle_measurements()
        np.testing.assert_almost_equal(
            _locations_for_angle_measurements, self.__expected_locations_for_angle_measurements_theta_45)

    def test_get_probabilities_for_angle_measurements(self):
        self.__object.create_density_functions()
        _probabilities_for_angle_measurements = self.__object.get_probabilities_for_angle_measurements(
            self.__measurements_to_compute_angle_probability)
        np.testing.assert_almost_equal(_probabilities_for_angle_measurements, 
                                       self.__expected_probabilities_for_angle_measurements)

    def test_run_measurement_update(self):
        self.reset_particles(np.array([0.1, 0.12, 0.15, 0.18, 0.2]))
        self.__object.create_density_functions()
        self.mock_get_undistorted_image()
        self.__object.run_measurement_update()
        np.testing.assert_almost_equal(
            self.__object.weights, self.__expected_weights_after_measurement_update)


if __name__ == '__main__':
    unittest.main()
    