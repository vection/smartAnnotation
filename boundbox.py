import cv2
import numpy as np
from math import sin, cos, atan, degrees, sqrt
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import re
from Point import Point



#
def min_value(x, y):
    return min(i for i in [x, y] if i is not None)


def max_value(x, y):
    return max(i for i in [x, y] if i is not None)

class BoundBox:
    def __init__(self, p1, p2, p3, p4, text_value=''):
        self._p1, self._p2, self._p3, self._p4 = p1,p2,p3,p4 #self.sort_corners(p1, p2, p3, p4)
        self._text_value = text_value


    def to_dict(self):
        return {'p1': self._p1, 'p2': self._p2, 'p3': self._p3, 'p4': self._p4, 'text': self.text_value}

    def sort_points(self):
        """
        sort the points of the box after transformations
        :return:
        """
        self._p1, self._p2, self._p3, self._p4 = self.sort_corners(self._p1, self._p2, self._p3, self._p4)

    @staticmethod
    def sort_corners(p1, p2, p3, p4):
        """
        sort the corners based on top-right, top-left, bottom-right, bottom left as p1, p2, p3 and p4
        :param p1: point 1
        :param p2: point 2
        :param p3: point 3
        :param p4: point 4
        :return: tuple of corners in sorted order
        TODO : implement a mechanism to check points are on the same line
        """

        # if any of the values is null return without sorting
        # this is to avoid None comparisons in case of void boxes
        if not any((p1.x, p1.y, p2.x, p2.y, p3.x, p3.y, p4.x, p4.y)):
            return p1, p2, p3, p4

        box = np.zeros((4, 2), dtype="int32")
        box[0] = [p1.x, p1.y]
        box[1] = [p2.x, p2.y]
        box[2] = [p3.x, p3.y]
        box[3] = [p4.x, p4.y]

        p_sum = box.sum(axis=1)
        p_diff = np.diff(box, axis=1)

        # points with max sum is bottom right and least sum is top left
        min_sum = min(p_sum)
        max_sum = max(p_sum)

        min_sum_index = np.where(p_sum == min_sum)[0]
        max_sum_index = np.where(p_sum == max_sum)[0]

        # points with least sum is top left
        if len(min_sum_index) > 1:
            # if more than one value with the same min sum exists we take the one with minimum y - x

            top_left_index = min_sum_index[0] if p_diff[min_sum_index[0]] < p_diff[min_sum_index[1]] \
                else min_sum_index[1]

        else:
            top_left_index = min_sum_index[0]

        if len(max_sum_index) > 1:
            # if more than one value with the same max sum exists we take the one with maximum y - x bottom right
            bottom_right_index = max_sum_index[0] if p_diff[max_sum_index[0]] > p_diff[max_sum_index[1]] \
                else max_sum_index[1]
        else:
            bottom_right_index = max_sum_index[0]

        top_left = box[top_left_index]
        bottom_right = box[bottom_right_index]

        remaining_box = np.delete(box, [top_left_index, bottom_right_index], axis=0)

        p_diff = np.diff(remaining_box, axis=1)

        # "y-x" is largest for bottom left and lowest for top right
        min_diff = min(p_diff)

        top_right_index = np.where(p_diff == min_diff)[0][0]
        # is one is top right the remaining one is top left
        bottom_left_index = 1 - top_right_index

        top_right = remaining_box[top_right_index]
        bottom_left = remaining_box[bottom_left_index]

        p1 = Point(int(top_left[0]), int(top_left[1]))
        p2 = Point(int(top_right[0]), int(top_right[1]))

        p3 = Point(int(bottom_right[0]), int(bottom_right[1]))
        p4 = Point(int(bottom_left[0]), int(bottom_left[1]))

        return p1, p2, p3, p4

    def __str__(self):
        return "{}".format(self._text_value)

    def __repr__(self):
        return "{}".format(self._text_value)

    def __add__(self, other):

        p1_x = min_value(self._p1.x, other.p1.x)
        p1_y = min_value(self._p1.y, other.p1.y)

        p1 = Point(p1_x, p1_y)

        p2_x = max_value(self._p2.x, other.p2.x)
        p2_y = min_value(self._p2.y, other.p2.y)

        p2 = Point(p2_x, p2_y)

        p3_x = max_value(self._p3.x, other.p3.x)
        p3_y = max_value(self._p3.y, other.p3.y)

        p3 = Point(p3_x, p3_y)

        p4_x = min_value(self._p4.x, other.p4.x)
        p4_y = max_value(self._p4.y, other.p4.y)

        p4 = Point(p4_x, p4_y)

        if self._text_value and other.text_value:
            new_text = self._text_value + ' ' + other.text_value

        else:
            new_text = self._text_value + other.text_value

        merged_box = BoundBox(p1, p2, p3, p4, new_text.strip())
        return merged_box


    @classmethod
    def create_box(cls, x1, y1, x2, y2, x3, y3, x4, y4, text_value=None):

        p1 = Point(x1, y1)
        p2 = Point(x2, y2)
        p3 = Point(x3, y3)
        p4 = Point(x4, y4)

        return cls(p1, p2, p3, p4, text_value)


    @classmethod
    def pytesseract_boxes(cls, data):
        """
        creates a list of boxes from pytesseract data
        :param data: result of pytesseract image_to_data
        :return: list of BoundBox object
        """

        box_list = []
        try:
            for i in range(len(data['level'])):
                # if data['text'][i]:
                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                corner_1 = Point(x, y)
                corner_2 = Point(x + w, y + h)
                box = cls.create_box_from_corners(corner_1, corner_2, data['text'][i])

                box_list.append(box)
        except TypeError as ee:
            if type(box_list) != dict:
                raise TypeError("the result of pytesseract should be passed as dictionary, please try "
                                "image_to_data(img, output_type=Output.DICT)")
            raise ee

        return box_list

    @classmethod
    def create_box_from_corners(cls, corner_1, corner_2, text_value=None):
        """
          corner_1  #########################
                    #                       #
                    #                       #
                    #                       #
                    #########################  corner_2
        :param text_value: text value inside the box
        :param corner_1: point object of corner 1
        :param corner_2: point object of corner 2
        :return: box object
        """

        p1 = corner_1
        p3 = corner_2
        p2 = Point(corner_2.x, corner_1.y)
        p4 = Point(corner_1.x, corner_2.y)

        return BoundBox(p1, p2, p3, p4, text_value)

    @classmethod
    def google_ocr_boxes(cls, data):
        """
        create a list of list of boxes from results from google ocr data
        here the result is a list of list because multiple pages can be parsed together
        and a list of boxes is created for each page
        :param data: google ocr response data as dict
        :return: list(list(boxes))
        """

        page_list = []
        google_response = data['responses']

        # fill zero values for places where google ocr omitted values
        for page in google_response:
            for annotation in page['textAnnotations']:
                for vertex in range(len(annotation['boundingPoly']['vertices'])):
                    if 'x' not in annotation['boundingPoly']['vertices'][vertex]:
                        annotation['boundingPoly']['vertices'][vertex]['x'] = 0

                    elif 'y' not in annotation['boundingPoly']['vertices'][vertex]:
                        annotation['boundingPoly']['vertices'][vertex]['y'] = 0

        for page in google_response:
            box_list = []
            text_annotations = page['textAnnotations'][1:]

            for annotation in text_annotations:
                box = cls.create_box(annotation['boundingPoly']['vertices'][0]['x'],
                                     annotation['boundingPoly']['vertices'][0]['y'],
                                     annotation['boundingPoly']['vertices'][1]['x'],
                                     annotation['boundingPoly']['vertices'][1]['y'],
                                     annotation['boundingPoly']['vertices'][2]['x'],
                                     annotation['boundingPoly']['vertices'][2]['y'],
                                     annotation['boundingPoly']['vertices'][3]['x'],
                                     annotation['boundingPoly']['vertices'][3]['y'],
                                     annotation['description'])
                box_list.append(box)
            page_list.append(box_list)

        return page_list

    @classmethod
    def labelimg_xml_boxes(cls, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        box_list = []

        for member in root.findall('object'):
            text = member[0].text
            corner_1 = Point(int(member[4][0].text), int(member[4][1].text))
            corner_2 = Point(int(member[4][2].text), int(member[4][3].text))
            box = cls.create_box_from_corners(corner_1, corner_2, text_value=text)
            box_list.append(box)

        return box_list

    @classmethod
    def box_from_contour(cls, countour):
        if len(countour) != 4:
            raise IndexError('need to approximate the contour to 4 sided polygon, currently contains {} '
                             'sides'.format(len(countour)))

        points = countour.reshape(4, 2)

        return cls.box_from_array(points)

    @classmethod
    def box_from_array(cls, array):
        if len(array) != 4:
            raise IndexError('need to approximate the contour to 4 sided polygon, currently contains {} '
                             'sides'.format(len(array)))

        p1, p2, p3, p4 = cls.array_to_points(array)

        return cls(p1, p2, p3, p4)

    @classmethod
    def from_center(cls, center_x, center_y, length, breadth, angle):
        """
        :param center_x: x coordinate of center
        :param center_y: y coordinate of center
        :param length: length of rectangle
        :param breadth: breadth of rectangle
        :param angle: angle in radian with respect to lower x axis
        :return:
        """

        p1 = Point(center_x - length / 2, center_y - breadth / 2)
        p2 = Point(center_x + length / 2, center_y - breadth / 2)
        p3 = Point(center_x + length / 2, center_y + breadth / 2)
        p4 = Point(center_x - length / 2, center_y + breadth / 2)

        box = cls(p1, p2, p3, p4)
        box.rotate(angle)

        return box

    def perspective_wrap(self, img):

        width_1 = self._p3 - self._p4
        width_2 = self._p2 - self._p1

        height_1 = self._p3 - self._p2
        height_2 = self._p4 - self._p1

        # take the maximum of the width and height for the new image

        max_width = max(int(width_1), int(width_2))
        max_height = max(int(height_1), int(height_2))

        # construct our destination points

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]], dtype="float32")

        # calculate the perspective transform matrix and warp
        # the perspective to grab the screen
        rect = np.zeros((4, 2), dtype="float32")
        rect[0] = [self._p1.x, self._p1.y]
        rect[1] = [self._p2.x, self._p2.y]
        rect[2] = [self._p3.x, self._p3.y]
        rect[3] = [self._p4.x, self._p4.y]

        m = cv2.getPerspectiveTransform(rect, dst)
        warp = cv2.warpPerspective(img, m, (max_width, max_height))

        return warp

    def scale_box(self, ratio_w, ratio_h):

        self._p1.x = round(self._p1.x / ratio_w)
        self._p2.x = round(self._p2.x * ratio_w)
        self._p3.x = round(self._p3.x * ratio_w)
        self._p4.x = round(self._p4.x / ratio_w)

        self._p1.y = round(self._p1.y / ratio_h)
        self._p2.y = round(self._p2.y / ratio_h)
        self._p3.y = round(self._p3.y * ratio_h)
        self._p4.y = round(self._p4.y * ratio_h)

    def draw_box(self, img, color1=(0, 255, 0), color2=(0, 0, 255), x_axis=0, y_axis=5, font_scale=0.5,
                 font_type=cv2.FONT_HERSHEY_SIMPLEX):
        points = np.array([[self._p1.x, self._p1.y], [self._p2.x, self._p2.y], [self._p3.x, self._p3.y],
                           [self._p4.x, self._p4.y]])
        cv2.polylines(img, np.int32([points]), True, color1, thickness=3)
        cv2.putText(img, self.text_value, (self.p1.x - x_axis, self.p1.y - y_axis), font_type, font_scale,
                    color2, 2)

        return img

    @property
    def p1(self):
        return self._p1

    @property
    def p2(self):
        return self._p2

    @property
    def p3(self):
        return self._p3

    @property
    def p4(self):
        return self._p4

    @p1.setter
    def p1(self, p):
        if isinstance(p, Point):
            raise TypeError("point should be an instance of Point Class")
        self._p1 = p

    @p2.setter
    def p2(self, p):
        if not isinstance(p, Point):
            raise TypeError("point should be an instance of Point Class")
        self._p2 = p

    @p3.setter
    def p3(self, p):
        if not isinstance(p, Point):
            raise TypeError("point should be an instance of Point Class")
        self._p3 = p

    @p4.setter
    def p4(self, p):
        if not isinstance(p, Point):
            raise TypeError("point should be an instance of Point Class")
        self._p4 = p

    @property
    def text_value(self):
        return self._text_value

    @text_value.setter
    def text_value(self, value):
        if not isinstance(value, str):
            raise TypeError("text value should be an instance of str Class")
        self._text_value = value

    @property
    def np_array(self):
        box = np.zeros((4, 2), dtype="int32")
        box[0] = [self._p1.x, self._p1.y]
        box[1] = [self._p2.x, self._p2.y]
        box[2] = [self._p3.x, self._p3.y]
        box[3] = [self._p4.x, self._p4.y]

        return box

    def plot_box(self):

        np_array = self.np_array
        array = np_array.tolist()
        # repeat the first point to create a 'closed loop'
        array.append(array[0])

        # create lists of x and y values
        xs, ys = zip(*array)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # start y axis from top
        plt.gca().invert_yaxis()

        # change marking of x axis to top
        ax.xaxis.tick_top()

        for i, p in enumerate(['p1', 'p2', 'p3', 'p4']):
            ax.annotate(p, (xs[i], ys[i]))

        plt.plot(xs, ys)
        plt.grid()
        plt.show()

    @staticmethod
    def horizontal_merge(box_1, box_2):
        """
        merge two boxes. the resulting box will have the left corners of box_1 and
        right corners of box_2
        :param box_1:
        :param box_2:
        :return:
        """

        p1 = box_1.p1
        p2 = box_2.p2
        p3 = box_2.p3
        p4 = box_1.p4

        new_text = box_1.text_value + ' ' + box_2.text_value

        try:
            merged_box = BoundBox(p1, p2, p3, p4, new_text.strip())
        except TypeError as err:
            if not box_1.p1.x or not box_1.p4.x:
                return box_2
            elif not box_2.p2.x or not box_2.p3.x:
                return box_1
            else:
                raise err

        return merged_box

    @staticmethod
    def vertical_merge(box_1, box_2):
        """
        Merge two boxes vertically
        :param box_1:
        :param box_2:
        :return:
        """

        x0 = min(box_1.p1.x, box_2.p1.x)
        y0 = min(box_1.p1.y, box_2.p1.y)
        x1 = max(box_1.p3.x, box_2.p3.x)
        y1 = max(box_1.p3.y, box_2.p3.y)

        p1 = Point(x0, y0)
        p2 = Point(x1, y0)
        p3 = Point(x1, y1)
        p4 = Point(x0, y1)

        if (box_2.p3.y + box_2.p1.y) / 2 < (box_1.p3.y + box_1.p1.y) / 2:
            new_text = box_2.text_value + ' ' + box_1.text_value
        else:
            new_text = box_1.text_value + ' ' + box_2.text_value

        try:
            bb_list = [box_1, box_2]
            merged_box = BoundBox(p1, p2, p3, p4, new_text.strip())
        except TypeError as err:
            if not box_1.p1.x or not box_1.p4.x:
                return box_2
            elif not box_2.p2.x or not box_2.p3.x:
                return box_1
            else:
                raise err

        return merged_box

    @staticmethod
    def compare_box_horizontally(box1, box2, dx, compare_angle=False, voverlap=False):
        """
        compare the boxes to check whether box2 is on the right side of box1 and they are close
            enough and parallel
        :param box1: left side box
        :param box2: right side box
        :param dx: ratio of distance between boxes to the height of text box
        :return: True or False whether they belong in the same line
        # TODO : In the current implementation the checking of y axis is not proper for slopes. need to change it
        """

        if dx <= 0:
            return False

        # check the distance between box1.p3 - box2.p4 and box1.p2 - box2.p1 are almost equal

        distance_threshold = abs(box1.p2 - box2.p3) / 10

        d1 = box1.p3 - box2.p4
        d2 = box1.p2 - box2.p1
        if abs(d1 - d2) > distance_threshold:
            return False

        # check difference between angles in degree
        if compare_angle:
            angle_diff_threshold = 5
            angle_diff = abs(degrees(box1.angle) - degrees(box2.angle))
            if angle_diff > angle_diff_threshold:
                return False

        box_height = box1.p2 - box1.p3
        # check they lie on the same x axis. We look for difference in y axis

        if voverlap:
            #             if not ((box1.p1.y <= box2.p1.y <= box1.p3.y or box1.p1.y <= box2.p3.y <= box1.p3.y) or
            #                     (box2.p1.y <= box1.p1.y <= box2.p3.y or box2.p1.y <= box1.p3.y <= box2.p3.y)):
            yoverlap = max(0, min(box1.p3.y, box2.p3.y) - max(box1.p1.y, box2.p1.y))
            box1_y = box1.p3.y - box1.p1.y
            box2_y = box2.p3.y - box2.p1.y
            pecent_overlap = max((yoverlap / box1_y), (yoverlap / box2_y))
            if pecent_overlap < voverlap:
                return False
        else:
            dy = box_height / 3
            if abs(box1.p3.y - box2.p4.y) > dy:
                return False

        # check distance between boxes
        distance_threshold = box_height * dx
        if box1.p2.x < box2.p1.x:
            distance_between_boxes = ((box2.p4.x - box1.p3.x) + (box2.p1.x - box1.p2.x)) / 2
            if distance_between_boxes > distance_threshold:
                return False

        elif box1.p1.x > box2.p2.x:
            return False

        return True

    @staticmethod
    def merge_boxes_inside(rects):
        points_inside = 0
        already_merged = []
        for ind, rec1 in enumerate(rects):
            for rec2 in rects:
                points = BoundBox.check_points_in_rectangle(rec1, rec2)
                if points >= 2 and rec1.text_value not in already_merged and rec1.text_value != rec2.text_value:
                    rec2.text_value += ' ' + rec1.text_value
                    already_merged.append(rec1.text_value)
        return rects

    @staticmethod
    def check_points_in_rectangle(rect1, rect2):
        a1, a2, a3, a4 = rect2.p1, rect2.p2, rect2.p3, rect2.p4
        points = 0
        if a1.y <= rect1.p1.y and rect1.p1.y <= a4.y and a1.x <= rect1.p1.x and rect1.p1.x <= a2.x:
            points += 1
        if a1.y <= rect1.p2.y and rect1.p2.y <= a4.y and a1.x <= rect1.p2.x and rect1.p2.x <= a2.x:
            points += 1
        if a1.y <= rect1.p3.y and rect1.p3.y <= a4.y and a1.x <= rect1.p3.x and rect1.p3.x <= a2.x:
            points += 1
        if a1.y <= rect1.p4.y and rect1.p4.y <= a4.y and a1.x <= rect1.p4.x and rect1.p4.x <= a2.x:
            points += 1
        return points

    @staticmethod
    def xy_distance(box1, box2):

        if min(box2.p3.x, box1.p3.x) - max(box2.p1.x, box1.p1.x) < 0:
            dx = min(abs(box1.p1.x - box2.p3.x), abs(box1.p3.x - box2.p1.x))
        else:
            dx = 0

        if min(box2.p3.y, box1.p3.y) - max(box2.p1.y, box1.p1.y) < 0:
            dy = min(abs(box1.p1.y - box2.p3.y), abs(box1.p3.y - box2.p1.y))
        else:
            dy = 0

        return dx, dy

    @staticmethod
    def x_overlap(box_1, box_2):

        box_1_xarea = box_1.p3.x - box_1.p1.x
        box_2_xarea = box_2.p3.x - box_2.p1.x

        if box_2_xarea <= 0 or box_1_xarea <= 0:
            return 0

        result = max(0, min(box_1.p3.x, box_2.p3.x) - max(box_1.p1.x, box_2.p1.x))

        # overlap_precent = 0.1
        # return overlap_precent*box_1_xarea <= result or overlap_precent*box_2_xarea <= result

        return result

    @staticmethod
    def compare_box_verticelly(box1, box2, dx):
        if BoundBox.x_overlap(box1, box2) <= 0:
            return False

        x, y = BoundBox.xy_distance(box1, box2)

        return dx >= y

    @staticmethod
    def bb_verticle_compare_numpy(a, b, dy=np.inf, xmatch=0.1):

        """
        a: list of box coordintes [[x0,y0,x3,y3], ...]
            where x0,y0 are the coordinates of the top left corner of a box, x3, y3 are the right bottom corner of the box
        b: the same list but shifted by N boxes
        """

        # get min/max of relevant coordinates to find intersection
        ab = np.stack([a, b], axis=0)
        abmax = np.max(ab, axis=0)
        abmin = np.min(ab, axis=0)
        ab_xarea = (ab[:, :, 2] - ab[:, :, 0])
        ab_min_xarea = np.min(ab_xarea, axis=0)

        xinter = abmin[:, 2] - abmax[:, 0]  # find if there's intersection in x axes

        xinter_area_prec = xinter / ab_min_xarea

        result = np.zeros_like(a)
        # y_dist = abmin[:,3] - abmax[:,1]
        # np.place(y_dist, y_dist < 0, 0)

        b_rolled = np.roll(b, 2, axis=1)  # find distance in the y axies
        y_distance = np.abs(a - b_rolled)
        y_distance_bottom_to_top = y_distance[:, 1]
        y_distance_top_to_bottom = y_distance[:, 3]

        y_distances_stacked = np.stack([y_distance_bottom_to_top, y_distance_top_to_bottom])
        y_min_distance = np.min(y_distances_stacked, axis=0)

        np.place(y_min_distance, y_min_distance < 0, 0)
        np.place(y_min_distance, y_min_distance > dy, -100)
        np.place(y_min_distance, xinter < 0, -50)
        np.place(y_min_distance, xinter_area_prec < xmatch, -50)

        return y_min_distance

    @staticmethod
    def do_y_match(a, dy=np.inf, xmatch=0.1):
        result_matrix = np.zeros([a.shape[0], a.shape[0]])  # + -1

        for offset in range(1, np.ceil(a.shape[0] / 2).astype(int) + 1):

            b = np.roll(a, offset, axis=0)

            result = BoundBox.bb_verticle_compare_numpy(a, b, dy=dy, xmatch=xmatch)
            for i, item in enumerate(result):
                if item >= 0:
                    result_matrix[i, i - offset] = 1

        return result_matrix

    @staticmethod
    def find_connected_compunents(result_matrix):
        d_graph = {i: set() for i in range(result_matrix.shape[0])}
        for i in range(result_matrix.shape[0]):
            for j in range(result_matrix.shape[1]):
                match = result_matrix[i][j]

                if match:
                    d_graph[i].add(j)
                    d_graph[j].add(i)

        def dfs(visited, graph, node):
            if node not in visited:

                visited.append(node)
                for neighbour in graph[node]:
                    dfs(visited, graph, neighbour)
                return visited

        d_graph = {k: list(v) for k, v in d_graph.items()}

        found = []
        res = []
        for i in range(len(d_graph)):
            if i not in found:
                res.append(dfs([], d_graph, i))
                found = found + res[-1]

        return res

    @staticmethod
    def bb_intersection_numpy_smallest(a, b):

        """
        a: list of box coordintes [[x0,y0,x3,y3], ...]
            where x0,y0 are the coordinates of the top left corner of a box, x3, y3 are the right bottom corner of the box
        b: the same list but shifted by N boxes
        """

        # get min/max of relevant coordinates to find intersection
        ab = np.stack([a, b], axis=0)
        abmax = np.max(ab, axis=0)
        abmax = np.roll(abmax, 2, axis=1)
        abmin = np.min(ab, axis=0)

        # find boxes areas
        boxAArea = np.abs((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))
        boxBArea = np.abs((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))
        minboxArea = np.stack([boxAArea, boxBArea]).min(axis=0)

        # find intersection area of boxes
        abmin_abmax = (abmin - abmax).clip(0)
        interArea = np.abs(abmin_abmax[:, 2] * abmin_abmax[:, 3])

        # find intersection area over smallest box
        result = interArea / minboxArea
        result[result == 0] = 0
        result = np.nan_to_num(result, copy=True, nan=0, posinf=0, neginf=0)

        return result

    @staticmethod
    def find_intersections_indecies(a):
        """
        this function returns which bounding boxes intersect and by how much
        coords: list of box coordintes [[x0,y0,x3,y3], ...]
            where x0,y0 are the coordinates of the top left corner of a box, x3, y3 are the right bottom corner of the box
        returns: a matrix of relative intesection area of smallest box
        """

        intersections = np.array([])
        intersections_precentage = np.array([])

        result_matrix = np.zeros([a.shape[0], a.shape[0]])

        for offset in range(1, np.floor(a.shape[0] / 2).astype(int) + 1):

            b = np.roll(a, offset, axis=0)

            ab_result = BoundBox.bb_intersection_numpy_smallest(a, b)

            for i, result in enumerate(ab_result):
                result_matrix[i][(i - offset) % a.shape[0]] = result

        return result_matrix


    @staticmethod
    def merge_box(box_list, dx=1, vx=30,merge_box_inside=True):
        """
        This function is used to merge similar kind of text in an image and create meaningful sentences
        :param box_list: list of box objects that need to be merged
        :param dx: ratio of distance between boxes to the height of text box, keep 1 as default
        :return: list of box objects where certain boxes are merged
        # TODO : The below implementation is not optimal or the best. Need to change it to clustering
        """
        is_number = lambda text: text.replace('.', '').replace(',', '').isdigit()

        # sort the boxlist by the the x value of point p1
        box_list.sort(key=lambda k: k.p1.x)

        # set same number of flags to zero
        process_flag = [False] * len(box_list)
        results = []

        # merge path mode
        merge_path = []
        while True:
            # if all boxes are processed stop the loop
            if all(process_flag):
                break

            # take the first unprocessed box as current box and set its flag as True
            current_box_index = process_flag.index(False)
            current_box = box_list[current_box_index]
            process_flag[current_box_index] = True

            # loop through the the unprocessed boxes
            for index, b in enumerate(box_list):

                # if it is already done skip it
                if process_flag[index]:
                    continue

                # compare the box 'b' horizontally with current box and check if they are near by
                if BoundBox.compare_box_horizontally(current_box, b, dx):
                    # if ignore_numbers and (is_number(current_box.text_value) or is_number(b.text_value)):
                    #    continue

                    current_box = BoundBox.horizontal_merge(current_box, b)
                    process_flag[index] = True
                    merge_path.append(b)

            results.append(current_box)

        if vx > 0:
            results.sort(key=lambda k: k.p1.y)
            box_list = results
            # set same number of flags to zero
            process_flag = [False] * len(box_list)
            results = []

            np_items = np.array([[box.p1.x, box.p1.y, box.p3.x, box.p3.y] for box in box_list])

            vymatrix = BoundBox.do_y_match(np_items, dy=vx, xmatch=0.1)

            connected = BoundBox.find_connected_compunents(vymatrix)

            for gi, group in enumerate(connected):
                current_box_index = group[0]
                current_box = box_list[current_box_index]

                for ii, other_box_index in enumerate(group):
                    if ii != 0:
                        b = box_list[other_box_index]
                        current_box = BoundBox.vertical_merge(current_box, b)
                        merge_path.append(b)

                results.append(current_box)

        # if merge_box_inside:
        #     results = BoundBox.merge_boxes_inside(results)

        if merge_box_inside:
            for i in range(4):

                # set same number of flags to zero
                results.sort(key=lambda k: k.p1.y)
                box_list = results
                process_flag = [False] * len(box_list)
                results = []

                np_items = np.array([[box.p1.x, box.p1.y, box.p3.x, box.p3.y] for box in box_list])
                intersection_matrix = BoundBox.find_intersections_indecies(np_items)
                connected = BoundBox.find_connected_compunents(intersection_matrix)

                for gi, group in enumerate(connected):
                    current_box_index = group[0]
                    current_box = box_list[current_box_index]

                    for ii, other_box_index in enumerate(group):
                        if ii != 0:
                            b = box_list[other_box_index]

                            current_box = BoundBox.vertical_merge(current_box, b)
                            merge_path.append(b)

                    results.append(current_box)



        results.sort(key=lambda k: k.p1.y)

        return results