#
# Emre Ay, April 2017
# Cyclops project: Localization system with an overhead camera
#

from scale_estimation import scale_estimation
import numpy as np
import argparse
import cv2
import yaml

class Keys:
    space = 32
    esc = 27
    enter = 10
    c = 99

class Scaler:
    def __init__(self, cam_src = 0, param_src = None):
        self.image = np.zeros((1,1,1), np.uint8)
        self.processed_image = np.zeros((1,1,1), np.uint8)
        self.roi = []
        self.done = False
        self.quit = False
        self.drag = False
        self.cropped = False
        self.undistort = False
        self.reference_width = 0.297
        self.vc = cv2.VideoCapture(cam_src)
        self.main_window = 'Scaler'
        self.cropped_window = 'Cropped'
        self.binary_window = 'Binary Image'
        self.result_window = 'Result'
        self.pixel_scale = None
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
        cv2.destroyWindow(self.main_window)
        cv2.destroyWindow(self.cropped_window)
        cv2.destroyWindow(self.binary_window)
        cv2.destroyWindow(self.result_window)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not self.drag:
            self.roi = [(x,y)]
            self.drag = True

        if event == cv2.EVENT_MOUSEMOVE and self.drag:
            temp_image = self.processed_image.copy()
            temp_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2RGB)
            cv2.rectangle(temp_image, self.roi[0], (x,y), (0, 255, 0), 2)
            cv2.imshow(self.main_window, temp_image)

        if event == cv2.EVENT_LBUTTONUP and self.drag:
            self.roi.append((x,y))
            temp_image = self.processed_image.copy()
            temp_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2RGB)
            cv2.rectangle(temp_image, self.roi[0], (x,y), (0, 255, 0), 2)
            cv2.imshow(self.main_window, temp_image)
            self.drag = False
            print('Selected region: {}'.format(self.roi))

        if event == cv2.EVENT_RBUTTONDOWN:
            self.roi = []
            self.drag = False
            self.cropped = False
            cv2.destroyWindow(self.cropped_window)
            cv2.destroyWindow(self.binary_window)
            cv2.destroyWindow(self.result_window)

    def process(self,image):
        self.processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.processed_image = cv2.GaussianBlur(self.processed_image, (7, 7), 0)
        cv2.namedWindow(self.main_window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.main_window, self.mouse_callback)

        if len(self.roi) == 2:
            temp_image = self.processed_image.copy()
            temp_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2RGB)
            cv2.rectangle(temp_image, self.roi[0], self.roi[1], (0, 0, 255), 2)
            cv2.imshow(self.main_window, temp_image)
        else:
            cv2.imshow(self.main_window, self.processed_image)

        k = cv2.waitKey(50)
        cropped_image = self.processed_image
        if k == Keys.space or self.cropped:
            self.cropped = True
            if len(self.roi) == 2:
                (r1,r2),(r3,r4) = self.roi
                y1 = min(r1,r3)
                y2 = max(r1,r3)
                x1 = min(r2,r4)
                x2 = max(r2,r4)
                cropped_image = cropped_image[x1:x2,y1:y2]
                cv2.namedWindow(self.cropped_window, cv2.WINDOW_AUTOSIZE)
                cv2.imshow(self.cropped_window, cropped_image)

        if k == Keys.c and self.cropped:
            th = int(np.mean(cropped_image))
            ret, thresh = cv2.threshold(cropped_image, th, 255, cv2.THRESH_BINARY)
            cv2.namedWindow(self.binary_window, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(self.binary_window, thresh)
            self.estimate_scale(thresh)

        if k == Keys.enter:
            if self.pixel_scale:
                self.destroy_windows()
                self.done = True
            else:
                print('Operation is not done yet.')

        if k == Keys.esc:
            self.destroy_windows()
            self.quit = True

    def estimate_scale(self, image):
            self.pixel_scale, color_image = scale_estimation(image, self.reference_width)
            cv2.namedWindow(self.result_window, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(self.result_window, color_image)

    def run(self):
        while not self.done and not self.quit:
            if self.vc.isOpened():
                _ , self.image = self.vc.read()
                if self.undistort:
                    cv2.undistort(self.image, self.camera_matrix, self.distortion)
                self.process(self.image)
        self.vc.release()
        return self.pixel_scale


class MemberInitializer:
    def __init__(self, cam_src = 0):
        self.color = None
        self.done = False
        self.quit = False
        self.drag = False
        self.cropped = False
        self.roi = []
        self.image = np.zeros((1,1,1), np.uint8)
        self.main_window = 'Add Member'
        self.cropped_window = 'Selected Area For Member Color'
        self.color_window = 'Selected Member Color'
        self.vc = cv2.VideoCapture(cam_src)

    def destroy_windows(self):
        cv2.destroyWindow(self.main_window)
        cv2.destroyWindow(self.cropped_window)
        cv2.destroyWindow(self.color_window)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not self.drag:
            self.roi = [(x,y)]
            self.drag = True

        if event == cv2.EVENT_MOUSEMOVE and self.drag:
            temp_image = self.image.copy()
            cv2.rectangle(temp_image, self.roi[0], (x,y), (0, 255, 0), 2)
            cv2.imshow(self.main_window, temp_image)

        if event == cv2.EVENT_LBUTTONUP and self.drag:
            self.roi.append((x,y))
            temp_image = self.image.copy()
            cv2.rectangle(temp_image, self.roi[0], (x,y), (0, 255, 0), 2)
            cv2.imshow(self.main_window, temp_image)
            self.drag = False

        if event == cv2.EVENT_RBUTTONDOWN:
            self.roi = []
            self.drag = False
            self.cropped = False
            cv2.destroyWindow(self.cropped_window)
            cv2.destroyWindow(self.color_window)

    def add_padding(self, image, padding=50):
        offset = padding/2
        # background image to show the selected area
        back = np.zeros((image.shape[0]+padding, image.shape[1]+padding, image.shape[2]), np.uint8)
        # overlaying cropped area on background image
        back[offset:offset+image.shape[0],offset:offset+image.shape[1]] = image
        return back

    def get_member_color(self):
        while not self.done and not self.quit:
            if self.vc.isOpened():
                _ , self.image = self.vc.read()
                cv2.namedWindow(self.main_window, cv2.WINDOW_AUTOSIZE)
                cv2.setMouseCallback(self.main_window, self.mouse_callback)
                k = cv2.waitKey(50)
                crop = self.image
                if k == Keys.space or self.cropped:
                    self.cropped = True
                    if len(self.roi) == 2:
                        (r1,r2),(r3,r4) = self.roi
                        y1 = min(r1,r3)
                        y2 = max(r1,r3)
                        x1 = min(r2,r4)
                        x2 = max(r2,r4)
                        crop = crop[x1:x2,y1:y2]
                        crop_pad = self.add_padding(crop)
                        cv2.namedWindow(self.cropped_window, cv2.WINDOW_AUTOSIZE)
                        cv2.imshow(self.cropped_window, crop_pad)

                if len(self.roi) == 2:
                    temp_image = self.image.copy()
                    cv2.rectangle(temp_image, self.roi[0], self.roi[1], (0, 0, 255), 2)
                    cv2.imshow(self.main_window, temp_image)
                else:
                    cv2.imshow(self.main_window, self.image)

                if k == Keys.c and self.cropped:
                    c = cv2.mean(crop)
                    color_image = crop
                    color_image[:] = (c[0],c[1],c[2])
                    color_pad = self.add_padding(color_image)
                    cv2.namedWindow(self.color_window, cv2.WINDOW_AUTOSIZE)
                    cv2.imshow(self.color_window, color_pad)
                    self.color = c
                    print('Average BGR for member id: {}'.format(self.color))

                if k == Keys.enter:
                    if self.color:
                        self.destroy_windows()
                        self.done = True
                    else:
                        print('Operation not done yet.')

                if k == Keys.esc:
                    self.destroy_windows()
                    self.quit = True
            else:
                print('vc not opened')
        self.vc.release()
        return self.color


class Member:
    members = int(0)
    def __init__(self):
        self.color = None
        self.initializer = MemberInitializer()
        Member.members += 1
        self.idx = Member.members
        # self.idx = self.__idx
    def get_color(self):
        self.color = self.initializer.get_member_color()
        return self.color
