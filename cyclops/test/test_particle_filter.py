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

        cls.__object = ParticleFilter(parameters_file = cls.__parameters_file_path)
        cls.__expected_loaded_paramaters = {"number_of_particles": 1000,
                                            "process_covariance": [1e-2, 1e-2, 1e-2],
                                            "measurement_covariance": [1e-2, 1e-2, 1e-2],
                                            "reference_distance": 5.0}
        cls.__expeced_number_of_particles = 1000
        cls.__expected_process_covariance = np.diag([1e-2, 1e-2, 1e-2])
        cls.__expected_measurement_covariance = np.diag([1e-2, 1e-2, 1e-2])
        cls.__expected_reference_distance = 5.0

    @classmethod
    def tearDownClass(cls):
        pass

    def test_init_with_empty_file(self):
        self.assertRaises(FileNotFoundError, ParticleFilter, tempfile.NamedTemporaryFile().name)

    def test_load_parameters(self):
        self.assertDictEqual(self.__object.parameters, self.__expected_loaded_paramaters)
    
    def test_set_variables(self):
        self.assertEqual(self.__object.number_of_particles, self.__expeced_number_of_particles)
        np.testing.assert_array_almost_equal(self.__object.process_covariance, self.__expected_process_covariance)
        np.testing.assert_array_almost_equal(self.__object.measurement_covariance, self.__expected_measurement_covariance)
        self.assertAlmostEqual(self.__object.reference_distance, self.__expected_reference_distance)


if __name__ == '__main__':
    unittest.main()
    