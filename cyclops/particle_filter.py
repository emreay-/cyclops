import os
import json

import numpy as np


class ParticleFilter(object):

    def __init__(self, parameters_file: str):
        self.parameters_file = parameters_file
        if self.check_parameters_file():
            self.load_parameters()
            self.set_variables()
        else:
            raise FileNotFoundError()
    
    def check_parameters_file(self):
        return os.path.exists(self.parameters_file)

    def load_parameters(self):
        with open(self.parameters_file, 'r') as handle:
            self.parameters = json.load(handle)

    def set_variables(self):
        self.number_of_particles = self.parameters['number_of_particles']
        self.process_covariance = np.diag(self.parameters['process_covariance'])
        self.measurement_covariance = np.diag(self.parameters['measurement_covariance'])
        self.reference_distance = self.parameters['reference_distance']
    

