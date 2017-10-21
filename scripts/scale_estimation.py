# These scripts are originated from Adrian Rosebrock's implementation available at
# PyImageSearch under the MIT License
# http://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/

# Copyright 2017 PyImageSearch
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
# to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from scipy.spatial import distance as dist
from imutils import perspective
import numpy as np
import imutils
import cv2


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def scale_estimation(image, reference_width):
    edge = cv2.Canny(image, 50, 100)
    edge = cv2.dilate(image, None, iterations=1)
    edge = cv2.erode(image, None, iterations=1)
    cnts = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # loop over the contours individually
    for c in cnts:
        if cv2.contourArea(c) < 100:
            continue
        # bounding box
        color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype='int')

        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        box = perspective.order_points(box)
        cv2.drawContours(color_image, [box.astype('int')], -1, (0, 255, 0), 2)
        for (x, y) in box:
            cv2.circle(color_image, (int(x), int(y)), 5, (0, 0, 255), -1)
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # midpoint visualization
        cv2.circle(color_image, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(color_image, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(color_image, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(color_image, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
        cv2.line(color_image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
            (255, 0, 255), 2)
        cv2.line(color_image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
            (255, 0, 255), 2)
        # distance computation
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # assuming that always the long edge measurement is supplied
        pixel_scale = max(dA,dB) / reference_width
        # compute the size of the object
        dimA = dA / pixel_scale
        dimB = dB / pixel_scale

        # draw the object sizes on the image
        cv2.putText(color_image, '{:.3f}m'.format(dimA),
            (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 255, 255), 2)
        cv2.putText(color_image, '{:.3f}m'.format(dimB),
            (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (170, 255, 255), 2)

    return pixel_scale, color_image
