import os
import sys
import unittest
import tempfile
import shutil

import numpy as np

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
        self.__expected_image_size = [480, 640]
        self.__expected_frame_translation = [240, 320]
        self.__expeced_number_of_particles = 1000
        self.__expected_process_covariance = np.diag([1e-2, 1e-2, 1e-2])
        self.__expected_measurement_covariance = np.diag([1e-2, 1e-2, 1e-2])
        self.__expected_inital_location_covariance = np.diag([1e-2, 1e-2, 1e-2])
        self.__expected_reference_distance = 5.0
        self.__expected_camera_matrix = np.reshape(np.array([696.019484, 0.000000, 349.918582, 
                                                             0.000000, 696.225875, 217.635331, 
                                                             0.000000, 0.000000, 1.000000]), (3, 3))
        self.__expected_distortion = np.array([0.259910, -0.975120, -0.021119, -0.007444, 0.000000])
        self.__start_location = (100., 100.)
        self.__expected_weights = np.ones((self.__expeced_number_of_particles))
        self.__expected_particle_thetas = np.array([ 5.0, 5.5, 6.0, 0.21681469, 0.71681469, 1.21681469])
        self.__particles_before_process = np.array([5.0, 5.5, 6.0, 6.5, 7.0, 7.5])
        
        self.__expected_captured_image = np.zeros((480, 640, 3), np.uint8)

    @classmethod
    def tearDownClass(cls):
        remove_directories_if_exist(cls.__temp_video_dir)

    def reset_particles(self):
        self.__object.particles = np.zeros((3, 6), dtype=float)
        self.__object.particles[:] = self.__particles_before_process
        self.__object.number_of_particles = 6

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
        self.assertListEqual(self.__object.image_size, self.__expected_image_size)
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
        self.reset_particles()
        self.__object.apply_mode_to_particle_thetas()
        np.testing.assert_array_almost_equal(self.__object.particles[2, :], self.__expected_particle_thetas)
    
    def test_capture_image(self):
        self.mock_video_capture()
        np.testing.assert_almost_equal(self.__object.captured_image, self.__expected_captured_image)

    def test_convert_particles_to_pixel_space(self):
        self.reset_particles()
        self.__object.convert_particles_to_pixel_space()

    def test_run_process_model(self):
        self.reset_particles()
        self.__object.run_process_model()
        self.assertRaises(AssertionError, np.testing.assert_almost_equal, 
                          self.__object.particles, self.__particles_before_process)

    def test_run_measurement_update(self):
        self.reset_particles()
        self.mock_get_undistorted_image()
        self.__object.run_measurement_update()

if __name__ == '__main__':
    unittest.main()
    