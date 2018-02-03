import os
import sys
import unittest
import tempfile
import shutil


class CyclopsUnitTestBase(unittest.TestCase):

    pass

def write_dummy_parameters_file(*args):
    for arg in args:
        with open(arg, 'w') as handle:
            handle.writelines('''{
    "number_of_particles": 1000,
    "process_covariance": [1e-2, 1e-2, 1e-2],
    "measurement_covariance": [1e-2, 1e-2, 1e-2],
    "reference_distance": 5.0
 }''')
 