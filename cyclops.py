#
# Emre Ay, April 2017
# Cyclops project: Localization system with an overhead camera
#

from scipy.spatial import distance as dist
from imutils import perspective
import numpy as np
import argparse
import imutils
import cv2
import yaml

class Keys:
    space = 1048608
    esc = 1048603
    enter = 1048586
    c = 1048675

# Class for estimating the scale for pixel/meter, assuming a perpendicular
# view on a planar scene. The estimation function is based on
# Adrian Rosebrock's implementation available at
# http://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/
class Scaler:
    def __init__(self, cam_src = 0, param_src = None):
        self.image = np.zeros((1,1,1), np.uint8)
        self.prc_image = np.zeros((1,1,1), np.uint8)
        self.roi = []
        self.done = False
        self.quit = False
        self.drag = False
        self.cropped = False
        self.undistort = False
        self.reference_width = 0.297
        self.keys = Keys()
        self.vc = cv2.VideoCapture(cam_src)
        self.main_window = "Scaler"
        self.cropped_window = "Cropped"
        self.binary_window = "Binary Image"
        self.result_window = "Result"
        self.pixel_scale = None
        if param_src:
            with open(param_src,"r") as stream:
                try:
                    camera_info = yaml.load(stream)
                except yaml.YAMLError as exception:
                    print exception
            if camera_info:
                self.undistort = True
                camera_matrix = camera_info["camera_matrix"]["data"]
                self.camera_matrix = np.reshape(np.array(camera_matrix),(3,3))
                distortion = camera_info["distortion_coefficients"]["data"]
                self.distortion = np.array(distortion)
                print "Camera calibration found. \nCamera matrix:\n{} \nDistortion:\n{}".format(camera_matrix,distortion)

    def midpoint(self, ptA, ptB):
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

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
            temp_img = self.prc_image.copy()
            temp_img = cv2.cvtColor(self.prc_image, cv2.COLOR_GRAY2RGB)
            cv2.rectangle(temp_img, self.roi[0], (x,y), (0, 255, 0), 2)
            cv2.imshow(self.main_window, temp_img)

        if event == cv2.EVENT_LBUTTONUP and self.drag:
            self.roi.append((x,y))
            temp_img = self.prc_image.copy()
            temp_img = cv2.cvtColor(self.prc_image, cv2.COLOR_GRAY2RGB)
            cv2.rectangle(temp_img, self.roi[0], (x,y), (0, 255, 0), 2)
            cv2.imshow(self.main_window, temp_img)
            self.drag = False
            print "Selected region: {}".format(self.roi)

        if event == cv2.EVENT_RBUTTONDOWN:
            self.roi = []
            self.drag = False
            self.cropped = False
            cv2.destroyWindow(self.cropped_window)
            cv2.destroyWindow(self.binary_window)
            cv2.destroyWindow(self.result_window)

    def process(self,img):
        self.prc_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.prc_image = cv2.GaussianBlur(self.prc_image, (7, 7), 0)
        cv2.namedWindow(self.main_window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.main_window, self.mouse_callback)

        if len(self.roi) == 2:
            temp_img = self.prc_image.copy()
            temp_img = cv2.cvtColor(self.prc_image, cv2.COLOR_GRAY2RGB)
            cv2.rectangle(temp_img, self.roi[0], self.roi[1], (0, 0, 255), 2)
            cv2.imshow(self.main_window, temp_img)
        else:
            cv2.imshow(self.main_window, self.prc_image)

        k = cv2.waitKey(50)
        cropped_image = self.prc_image
        if k == self.keys.space or self.cropped:
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

        if k == self.keys.c and self.cropped:
            th = int(np.mean(cropped_image))
            ret, thresh = cv2.threshold(cropped_image, th, 255, cv2.THRESH_BINARY)
            cv2.namedWindow(self.binary_window, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(self.binary_window, thresh)
            self.estimate_scale(thresh)

        if k == self.keys.enter:
            if self.pixel_scale:
                self.destroy_windows()
                self.done = True
            else:
                print "Operation is not done yet."

        if k == self.keys.esc:
            self.destroy_windows()
            self.quit = True

    def estimate_scale(self,img):
        edge = cv2.Canny(img, 50, 100)
        edge = cv2.dilate(img, None, iterations=1)
        edge = cv2.erode(img, None, iterations=1)
        cnts = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        # loop over the contours individually
        for c in cnts:
            if cv2.contourArea(c) < 100:
                continue
            # bounding box
            converted = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")

            # order the points in the contour such that they appear
            # in top-left, top-right, bottom-right, and bottom-left
            # order, then draw the outline of the rotated bounding
            # box
            box = perspective.order_points(box)
            cv2.drawContours(converted, [box.astype("int")], -1, (0, 255, 0), 2)
            for (x, y) in box:
                cv2.circle(converted, (int(x), int(y)), 5, (0, 0, 255), -1)
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = self.midpoint(tl, tr)
            (blbrX, blbrY) = self.midpoint(bl, br)
            (tlblX, tlblY) = self.midpoint(tl, bl)
            (trbrX, trbrY) = self.midpoint(tr, br)

            # midpoint visualization
            cv2.circle(converted, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(converted, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(converted, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(converted, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
            cv2.line(converted, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                (255, 0, 255), 2)
            cv2.line(converted, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                (255, 0, 255), 2)
            # distance computation
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            if self.pixel_scale is None:
                # assuming that always the long edge measurement is supplied
                self.pixel_scale = max(dA,dB) / self.reference_width
            # compute the size of the object
            dimA = dA / self.pixel_scale
            dimB = dB / self.pixel_scale

        	# draw the object sizes on the image
            cv2.putText(converted, "{:.3f}m".format(dimA),
                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 255, 255), 2)
            cv2.putText(converted, "{:.3f}m".format(dimB),
                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (170, 255, 255), 2)

            cv2.namedWindow(self.result_window, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(self.result_window, converted)

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
        self.keys = Keys()
        self.roi = []
        self.image = np.zeros((1,1,1), np.uint8)
        self.main_window = "Add Member"
        self.cropped_window = "Selected Area For Member Color"
        self.color_window = "Selected Member Color"
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
            temp_img = self.image.copy()
            cv2.rectangle(temp_img, self.roi[0], (x,y), (0, 255, 0), 2)
            cv2.imshow(self.main_window, temp_img)

        if event == cv2.EVENT_LBUTTONUP and self.drag:
            self.roi.append((x,y))
            temp_img = self.image.copy()
            cv2.rectangle(temp_img, self.roi[0], (x,y), (0, 255, 0), 2)
            cv2.imshow(self.main_window, temp_img)
            self.drag = False

        if event == cv2.EVENT_RBUTTONDOWN:
            self.roi = []
            self.drag = False
            self.cropped = False
            cv2.destroyWindow(self.cropped_window)
            cv2.destroyWindow(self.color_window)

    def add_padding(self, img, padding=50):
        offset = padding/2
        # background image to show the selected area
        back = np.zeros((img.shape[0]+padding, img.shape[1]+padding, img.shape[2]), np.uint8)
        # overlaying cropped area on background image
        back[offset:offset+img.shape[0],offset:offset+img.shape[1]] = img
        return back

    def get_member_color(self):
        while not self.done and not self.quit:
            if self.vc.isOpened():
                _ , self.image = self.vc.read()
                cv2.namedWindow(self.main_window, cv2.WINDOW_AUTOSIZE)
                cv2.setMouseCallback(self.main_window, self.mouse_callback)
                k = cv2.waitKey(50)
                crop = self.image
                if k == self.keys.space or self.cropped:
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
                    temp_img = self.image.copy()
                    cv2.rectangle(temp_img, self.roi[0], self.roi[1], (0, 0, 255), 2)
                    cv2.imshow(self.main_window, temp_img)
                else:
                    cv2.imshow(self.main_window, self.image)

                if k == self.keys.c and self.cropped:
                    c = cv2.mean(crop)
                    color_img = crop
                    color_img[:] = (c[0],c[1],c[2])
                    color_pad = self.add_padding(color_img)
                    cv2.namedWindow(self.color_window, cv2.WINDOW_AUTOSIZE)
                    cv2.imshow(self.color_window, color_pad)
                    self.color = c
                    print "Average BGR for member id: {}".format(self.color)

                if k == self.keys.enter:
                    if self.color:
                        self.destroy_windows()
                        self.done = True
                    else:
                        print "Operation not done yet."

                if k == self.keys.esc:
                    self.destroy_windows()
                    self.quit = True
            else:
                print "vc not opened"
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
