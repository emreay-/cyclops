#
# Emre Ay, April 2017
# Cyclops project: Localization system with an overhead camera
#

import argparse
from typing import Callable

import numpy as np
import cv2
import yaml

from cyclops.scale_estimation import scale_estimation


class Keys:
    space = 32
    esc = 27
    enter = 10
    c = 99


class VideoCaptureInterface(object):

    def __init__(self, cam_src = 0, main_window_name: str = 'Main'):
        self.main_window = main_window_name
        self.roi = []
        self.quit = False
        self.is_done = False
        self.is_dragging = False
        self.is_cropped = False
        self.image = np.zeros((1,1,1), np.uint8)
        self.processed_image = np.zeros((1,1,1), np.uint8)
        self.video_capture = cv2.VideoCapture(cam_src)

    def destroy_windows(self):
        cv2.destroyWindow(self.main_window)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not self.is_dragging:
            self.start_dragging(x, y)

        if event == cv2.EVENT_MOUSEMOVE and self.is_dragging:
            self.dragging_process(x, y)

        if event == cv2.EVENT_LBUTTONUP and self.is_dragging:
            self.stop_dragging(x, y)

        if event == cv2.EVENT_RBUTTONDOWN:
            self.right_click_process(x, y)
    
    def start_dragging(self, x, y):
        self.roi = [(x,y)]
        self.is_dragging = True

    def dragging_process(self, x, y):
        temp_image = self.processed_image.copy()
        cv2.rectangle(temp_image, self.roi[0], (x,y), (0, 255, 0), 2)
        cv2.imshow(self.main_window, temp_image)
    
    def stop_dragging(self, x, y):
        self.roi.append((x,y))
        temp_image = self.processed_image.copy()
        cv2.rectangle(temp_image, self.roi[0], (x,y), (0, 255, 0), 2)
        cv2.imshow(self.main_window, temp_image)
        self.is_dragging = False
    
    def right_click_process(self, x, y):
        self.roi = []
        self.is_dragging = False
        self.is_cropped = False

    def run_loop(self, key_processor: Callable):
        self.handle_main_window()

        while not self.is_done and not self.quit:
            if self.video_capture.isOpened():
                self.capture_image()
                self.process_image()
                key = self.wait_key()

                if self.is_region_selected():
                    self.show_selected_region_on_processed_image()
                else:
                    self.show_processed_image()
                
                key_processor(key)

            else:
                print('VideoCapture not opened')

        self.video_capture.release()

    def handle_main_window(self):
        self.create_main_window()
        self.set_mouse_callback()

    def create_main_window(self):
        cv2.namedWindow(self.main_window, cv2.WINDOW_AUTOSIZE)

    def set_mouse_callback(self):
        cv2.setMouseCallback(self.main_window, self.mouse_callback)

    def capture_image(self):
        _ , self.image = self.video_capture.read()

    def process_image(self):
        self.processed_image = self.image

    def wait_key(self):
        return cv2.waitKey(50)

    def is_region_selected(self):
        return len(self.roi) == 2

    def show_selected_region_on_processed_image(self):
        temp_image = self.processed_image.copy()
        cv2.rectangle(temp_image, self.roi[0], self.roi[1], (0, 0, 255), 2)
        cv2.imshow(self.main_window, temp_image)

    def show_processed_image(self):
        cv2.imshow(self.main_window, self.processed_image)

    def get_top_left_bottom_right_points_from_roi(self):
        (r1,r2),(r3,r4) = self.roi
        x1 = min(r1,r3)
        x2 = max(r1,r3)
        y1 = min(r2,r4)
        y2 = max(r2,r4)
        return (x1, y1), (x2, y2)
    
    @staticmethod
    def add_padding(image, padding=50):
        offset = int(padding / 2)
        image_with_padding = np.zeros((image.shape[0]+padding, image.shape[1]+padding, image.shape[2]), np.uint8)
        image_with_padding[offset:offset+image.shape[0],offset:offset+image.shape[1]] = image
        return image_with_padding

    def process_escape_key(self, key):
        if key == Keys.esc:
            self.destroy_windows()
            self.quit = True


class Scaler(VideoCaptureInterface):
    
    def __init__(self, cam_src = 0, param_src: str = None):
        super().__init__(cam_src, main_window_name = 'Scaler')
        self.undistort = False
        self.reference_width = 0.297
        self.cropped_window = 'Cropped'
        self.binary_window = 'Binary Image'
        self.result_window = 'Result'
        self.pixel_scale = None
        self.load_camera_parameters_if_given(param_src)

    def load_camera_parameters_if_given(self, param_src):
        if param_src:
            with open(param_src,'r') as stream:
                try:
                    camera_info = yaml.load(stream)
                except yaml.YAMLError as exception:
                    print(exception)
            if camera_info:
                self.undistort = True
                camera_matrix = camera_info['camera_matrix']['data']
                self.camera_matrix = np.reshape(np.array(camera_matrix),(3,3))
                distortion = camera_info['distortion_coefficients']['data']
                self.distortion = np.array(distortion)
                print('Camera calibration found. \nCamera matrix:\n{} \nDistortion:\n{}'.format(camera_matrix,distortion))

    def destroy_windows(self):
        super().destroy_windows()
        self.destroy_non_main_windows()

    def destroy_non_main_windows(self):
        cv2.destroyWindow(self.cropped_window)
        cv2.destroyWindow(self.binary_window)
        cv2.destroyWindow(self.result_window)

    def dragging_process(self, x, y):
        temp_image = self.processed_image.copy()
        temp_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2RGB)
        cv2.rectangle(temp_image, self.roi[0], (x,y), (0, 255, 0), 2)
        cv2.imshow(self.main_window, temp_image)

    def stop_dragging(self, x, y):
        self.roi.append((x,y))
        temp_image = self.processed_image.copy()
        temp_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2RGB)
        cv2.rectangle(temp_image, self.roi[0], (x,y), (0, 255, 0), 2)
        cv2.imshow(self.main_window, temp_image)
        self.is_dragging = False
        print('Selected region: {}'.format(self.roi))

    def right_click_process(self, x, y):
        super().right_click_process(x, y)
        self.destroy_non_main_windows()

    def process_image(self):
        if self.undistort:
            self.processed_image = cv2.undistort(self.image, self.camera_matrix, self.distortion)
        self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        self.processed_image = cv2.GaussianBlur(self.processed_image, (7, 7), 0)

    def run(self):
        self.run_loop(self.key_processor)
        return self.pixel_scale

    def key_processor(self, key):
        if key == Keys.space or self.is_cropped:
            self.is_cropped = True
            if self.is_region_selected():
                (x1, y1), (x2, y2) = self.get_top_left_bottom_right_points_from_roi()
                self.cropped_image = self.processed_image[y1:y2, x1:x2]
                cv2.namedWindow(self.cropped_window, cv2.WINDOW_AUTOSIZE)
                cv2.imshow(self.cropped_window, self.cropped_image)

        if key == Keys.c and self.is_cropped:
            threshold = int(np.mean(self.cropped_image))
            _, binary_image = cv2.threshold(self.cropped_image, threshold, 255, cv2.THRESH_BINARY)
            cv2.namedWindow(self.binary_window, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(self.binary_window, binary_image)
            self.estimate_scale(binary_image)

        if key == Keys.enter:
            if self.pixel_scale:
                self.destroy_windows()
                self.is_done = True
            else:
                print('Operation is not done yet.')

        self.process_escape_key(key)

    def estimate_scale(self, image):
            self.pixel_scale, color_image = scale_estimation(image, self.reference_width)
            cv2.namedWindow(self.result_window, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(self.result_window, color_image)


class MemberInitializer(VideoCaptureInterface):

    def __init__(self, cam_src = 0):
        super().__init__(cam_src, main_window_name = 'Member Initializer')
        self.cropped_window = 'Selected Area For Member Color'
        self.color_window = 'Selected Member Color'

    def destroy_windows(self):
        super().destroy_windows()
        self.destroy_non_main_windows()

    def destroy_non_main_windows(self):
        cv2.destroyWindow(self.cropped_window)
        cv2.destroyWindow(self.color_window)

    def right_click_process(self, x, y):
        self.destroy_non_main_windows()

    def get_member_color(self):
        self.run_loop(self.color_initialization_key_processor)
        return self.color

    def initialize_member_location(self):
        self.run_loop(self.location_initialization_key_processor)
        return self.init_location
    
    def color_initialization_key_processor(self, key):
        if key == Keys.space or self.is_cropped:
            self.is_cropped = True
            if len(self.roi) == 2:
                (x1, y1), (x2, y2) = self.get_top_left_bottom_right_points_from_roi()
                self.cropped_image = self.image[y1:y2, x1:x2]
                cv2.namedWindow(self.cropped_window, cv2.WINDOW_AUTOSIZE)
                cv2.imshow(self.cropped_window, self.add_padding(self.cropped_image))

        if key == Keys.c and self.is_cropped:
            mean_color = cv2.mean(self.cropped_image)
            mean_color_image = self.cropped_image
            self.color = (mean_color[0], mean_color[1], mean_color[2])
            mean_color_image[:] = self.color
            cv2.namedWindow(self.color_window, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(self.color_window, self.add_padding(mean_color_image))
            print('Average BGR for member id: {}'.format(self.color))

        if key == Keys.enter:
            if self.color:
                self.destroy_windows()
                self.is_done = True
            else:
                print('Operation not done yet.')

        self.process_escape_key(key)

    def location_initialization_key_processor(self, key):
        if key == Keys.enter:
            if len(self.roi) == 2:
                (x1, y1), (x2, y2) = self.get_top_left_bottom_right_points_from_roi()
                self.init_location = ((x1 + x2) / 2.0, (y1 + y2) / 2.0) 
                self.destroy_windows()
                self.is_done = True
            
            else:
                print('Operation not done yet.')

        self.process_escape_key(key)


class Member:
    members = int(0)

    def __init__(self):
        self._color = None
        Member.members += 1
        self._id = Member.members

    def initialize_color(self):
        self._color = MemberInitializer().get_member_color()
    
    def initialize_location(self):
        self._init_location = MemberInitializer().initialize_member_location()

    @property
    def color(self):
        return self._color

    @property
    def id(self):
        return self._id

    @property
    def initial_location(self):
        return self._init_location
